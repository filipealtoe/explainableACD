import argparse
import json
import logging
import os
import re
import warnings
from collections.abc import Generator
from datetime import datetime
from typing import Any

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import plotly.graph_objects as go
import polars as pl
import pyarrow.parquet as pq
from sentence_transformers import SentenceTransformer
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import IncrementalPCA
from sklearn.feature_extraction.text import CountVectorizer
from src.utils.paths import get_repo_root, resolve_path

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
logging.getLogger("mlflow.models.model").setLevel(logging.ERROR)
logging.getLogger("mlflow.utils.environment").setLevel(logging.ERROR)


def ensure_dir(d: str | os.PathLike[str]) -> None:
    os.makedirs(d, exist_ok=True)


def configure_experiment(name: str, artifact_root: str) -> None:
    try:
        mlflow.set_experiment(name, artifact_location=artifact_root)
        return
    except TypeError:
        pass
    exp = mlflow.get_experiment_by_name(name)
    if exp is None:
        try:
            mlflow.create_experiment(name, artifact_location=artifact_root)
        except TypeError:
            mlflow.create_experiment(name)
    mlflow.set_experiment(experiment_name=name)


def parse_timestamp(df: pl.DataFrame, created_at_col: str = "created_at", date_col: str = "date", time_col: str = "time") -> pl.Series:
    if created_at_col in df.columns:
        return df[created_at_col].str.replace(r" [A-Z]{3,4}$", "").str.to_datetime(format="%Y-%m-%d %H:%M:%S", strict=False)
    elif date_col in df.columns and time_col in df.columns:
        return pl.concat_str([df[date_col].cast(pl.Utf8), pl.lit(" "), df[time_col].cast(pl.Utf8)]).str.to_datetime(format="%Y-%m-%d %H:%M:%S", strict=False)
    else:
        return df.to_series(0).str.to_datetime(format="%Y-%m-%d %H:%M:%S", strict=False)


def clean_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.lower()
    s = re.sub(r"http\S+", " ", s)
    s = re.sub(r"@\w+", " ", s)
    s = re.sub(r"#", "", s)
    s = re.sub(r"[^\x00-\x7F]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def embeddings_cache_path(output_dir: str, model_name: str, chunk_idx: int) -> str:
    safe_model = model_name.replace("/", "_")
    return os.path.join(output_dir, "embeddings", f"emb_{safe_model}_chunk{chunk_idx:05d}.npy")


def save_embeddings(emb: np.ndarray[tuple[int, int], np.dtype[np.float32]], path: str) -> None:
    ensure_dir(os.path.dirname(path))
    np.save(path, emb)
    logger.info("Saved embeddings %s shape=%s", path, emb.shape)


def load_embeddings(path: str) -> np.ndarray[tuple[int, int], np.dtype[np.float32]]:
    loaded = np.load(path)
    assert isinstance(loaded, np.ndarray)
    return loaded


def read_data_chunks(file_path: str, chunk_size: int) -> Generator[pl.DataFrame]:
    if file_path.endswith(".parquet"):
        parquet_file = pq.ParquetFile(file_path)
        for batch in parquet_file.iter_batches(batch_size=chunk_size):
            yield pl.from_arrow(batch)
    else:
        df = pl.read_csv(file_path, infer_schema_length=0)
        for i in range(0, len(df), chunk_size):
            yield df.slice(i, chunk_size)


class StreamingBERTopicPipeline:
    def __init__(self,
                 output_dir: str,
                 embedding_model_name: str = "all-MiniLM-L6-v2",
                 pca_dim: int = 64,
                 n_clusters: int = 100,
                 batch_size_embed: int = 256):
        self.output_dir = output_dir
        ensure_dir(self.output_dir)
        ensure_dir(os.path.join(self.output_dir, "embeddings"))

        self.embedding_model_name = embedding_model_name
        device = "mps" if hasattr(__import__('torch').backends, 'mps') and __import__('torch').backends.mps.is_available() else "cpu"
        self.embedder = SentenceTransformer(self.embedding_model_name, device=device)
        logger.info("Using device: %s", device)
        self.pca_dim = pca_dim
        self.n_clusters = n_clusters
        self.batch_size_embed = batch_size_embed

        self.ipca: IncrementalPCA | None = None
        self.kmeans: MiniBatchKMeans | None = None

        self.doc_index: list[str] = []
        self.docs: list[str] = []
        self.timestamps: list[datetime | None] = []
        self.topic_assignments: list[int] = []

        self.state_path = os.path.join(self.output_dir, "pipeline_state.joblib")
        self.meta_path = os.path.join(self.output_dir, "pipeline_meta.json")

    def save_state(self) -> None:
        logger.info("Saving pipeline state...")
        state = {
            "embedding_model_name": self.embedding_model_name,
            "pca_dim": self.pca_dim,
            "n_clusters": self.n_clusters,
            "batch_size_embed": self.batch_size_embed,
            "doc_index": self.doc_index,
            "timestamps": [ts.isoformat() if ts is not None else None for ts in self.timestamps],
            "topic_assignments": self.topic_assignments,
            "docs_len": len(self.docs)
        }
        docs_path = os.path.join(self.output_dir, "docs.txt")
        with open(docs_path, "w", encoding="utf-8") as fh:
            for d in self.docs:
                fh.write(d.replace("\n", " ") + "\n")

        with open(self.meta_path, "w", encoding="utf-8") as fh:
            json.dump(state, fh)

        model_objs = {"ipca": self.ipca, "kmeans": self.kmeans}
        joblib.dump(model_objs, self.state_path)
        logger.info("Saved pipeline metadata to %s and models to %s", self.meta_path, self.state_path)

    def load_state(self) -> None:
        if not os.path.exists(self.meta_path) or not os.path.exists(self.state_path):
            raise FileNotFoundError("Pipeline state not found in output_dir.")
        logger.info("Loading pipeline state from %s ...", self.output_dir)
        with open(self.meta_path, encoding="utf-8") as fh:
            state = json.load(fh)
        self.embedding_model_name = state["embedding_model_name"]
        self.embedder = SentenceTransformer(self.embedding_model_name)
        self.pca_dim = state["pca_dim"]
        self.n_clusters = state["n_clusters"]
        self.batch_size_embed = state.get("batch_size_embed", self.batch_size_embed)

        docs_path = os.path.join(self.output_dir, "docs.txt")
        self.docs = []
        with open(docs_path, encoding="utf-8") as fh:
            for line in fh:
                self.docs.append(line.strip())

        self.doc_index = state["doc_index"]
        self.timestamps = [datetime.fromisoformat(x) if x is not None else None for x in state["timestamps"]]
        self.topic_assignments = state["topic_assignments"]

        models = joblib.load(self.state_path)
        self.ipca = models.get("ipca")
        self.kmeans = models.get("kmeans")
        logger.info("Loaded pipeline: %d docs, %d topics", len(self.docs), len(set(self.topic_assignments)))

    def embed_texts(self, texts: list[str]) -> np.ndarray[tuple[int, int], np.dtype[np.float32]]:
        return self.embedder.encode(texts, batch_size=self.batch_size_embed, show_progress_bar=False, convert_to_numpy=True)

    def initial_fit_stream(self,
                           csv_path: str,
                           text_col: str = "tweet",
                           id_col: str = "id",
                           created_at_col: str = "created_at",
                           chunk_size: int = 20000,
                           max_chunks: int | None = None) -> None:
        logger.info("Starting initial fit stream from %s", csv_path)

        self.ipca = IncrementalPCA(n_components=self.pca_dim)
        self.kmeans = MiniBatchKMeans(n_clusters=self.n_clusters, batch_size=1024)

        chunk_idx = 0

        for chunk in read_data_chunks(csv_path, chunk_size):
            chunk_idx += 1
            if max_chunks is not None and chunk_idx > max_chunks:
                break

            logger.info("Processing chunk %d", chunk_idx)
            chunk = chunk.filter(pl.col(text_col).is_not_null())
            chunk = chunk.with_columns(pl.col(text_col).map_elements(clean_text, return_dtype=pl.Utf8).alias("cleaned"))
            chunk = chunk.with_columns(parse_timestamp(chunk, created_at_col=created_at_col).alias("timestamp"))

            docs = chunk["cleaned"].to_list()
            ids = chunk[id_col].to_list()
            ts_list = chunk["timestamp"].to_list()

            batch_emb = self.embed_texts(docs).astype(np.float64, copy=False)
            save_path = embeddings_cache_path(self.output_dir, self.embedding_model_name, chunk_idx)
            save_embeddings(batch_emb, save_path)

            try:
                self.ipca.partial_fit(batch_emb)
            except Exception as e:
                logger.warning("IPCA partial_fit failed: %s", e)
                self.ipca.fit(batch_emb)

            emb_reduced = self.ipca.transform(batch_emb).astype(np.float64, copy=False)

            try:
                self.kmeans.partial_fit(emb_reduced)
            except Exception as e:
                logger.warning("KMeans partial_fit failed: %s", e)
                self.kmeans.fit(emb_reduced)

            labels = self.kmeans.predict(emb_reduced).tolist()

            for i, doc in enumerate(docs):
                self.doc_index.append(ids[i])
                self.docs.append(doc)
                self.timestamps.append(ts_list[i])
                self.topic_assignments.append(int(labels[i]))

            logger.info("Chunk %d processed: %d docs appended (total %d)", chunk_idx, len(docs), len(self.docs))

        self.save_state()
        logger.info("Initial streaming fit complete. Total docs: %d", len(self.docs))

    def incremental_update(self,
                           csv_path: str,
                           text_col: str = "tweet",
                           id_col: str = "id",
                           created_at_col: str = "created_at",
                           chunk_size: int = 5000,
                           update_keywords: bool = True,
                           reassign_on_update: bool = False) -> None:
        if self.ipca is None or self.kmeans is None:
            raise RuntimeError("Pipeline models not initialized. Call initial_fit_stream or load_state first.")

        added = 0
        existing_chunks = len([f for f in os.listdir(os.path.join(self.output_dir, "embeddings")) if f.endswith(".npy")]) \
                          if os.path.exists(os.path.join(self.output_dir, "embeddings")) else 0
        chunk_idx = existing_chunks

        for chunk in read_data_chunks(csv_path, chunk_size):
            chunk_idx += 1
            chunk = chunk.filter(pl.col(text_col).is_not_null())
            chunk = chunk.with_columns(pl.col(text_col).map_elements(clean_text, return_dtype=pl.Utf8).alias("cleaned"))
            chunk = chunk.with_columns(parse_timestamp(chunk, created_at_col=created_at_col).alias("timestamp"))

            docs = chunk["cleaned"].to_list()
            ids = chunk[id_col].to_list()
            ts_list = chunk["timestamp"].to_list()

            batch_emb = self.embed_texts(docs).astype(np.float64, copy=False)
            save_path = embeddings_cache_path(self.output_dir, self.embedding_model_name, chunk_idx)
            save_embeddings(batch_emb, save_path)

            try:
                self.ipca.partial_fit(batch_emb)
            except Exception as e:
                logger.warning("IPCA partial_fit failed on update: %s", e)

            emb_reduced = self.ipca.transform(batch_emb).astype(np.float64, copy=False)

            try:
                self.kmeans.partial_fit(emb_reduced)
            except Exception as e:
                logger.warning("KMeans partial_fit failed on update: %s", e)

            labels = self.kmeans.predict(emb_reduced).tolist()

            for i in range(len(docs)):
                self.doc_index.append(ids[i])
                self.docs.append(docs[i])
                self.timestamps.append(ts_list[i])
                self.topic_assignments.append(int(labels[i]))
                added += 1

            logger.info("Updated with chunk %d: added %d docs (total %d)", chunk_idx, len(docs), len(self.docs))

        logger.info("Incremental update finished: added %d docs", added)
        if reassign_on_update:
            self.reassign_all_documents()

        self.save_state()

    def assign_documents(self,
                         csv_path: str,
                         text_col: str = "tweet",
                         id_col: str = "id",
                         created_at_col: str = "created_at",
                         chunk_size: int = 2000) -> pl.DataFrame:
        if self.ipca is None or self.kmeans is None:
            raise RuntimeError("Pipeline models not initialized. Call load_state first.")

        result_chunks = []
        for chunk in read_data_chunks(csv_path, chunk_size):
            chunk = chunk.filter(pl.col(text_col).is_not_null())
            chunk = chunk.with_columns(pl.col(text_col).map_elements(clean_text, return_dtype=pl.Utf8).alias("cleaned"))
            chunk = chunk.with_columns(parse_timestamp(chunk, created_at_col=created_at_col).alias("timestamp"))

            docs = chunk["cleaned"].to_list()
            ids = chunk[id_col].to_list()
            ts_list = chunk["timestamp"].to_list()

            batch_emb = self.embed_texts(docs)
            emb_reduced = self.ipca.transform(batch_emb)
            labels = self.kmeans.predict(emb_reduced).tolist()

            result_df = pl.DataFrame({
                id_col: ids,
                "timestamp": ts_list,
                "cleaned_text": docs,
                "topic": labels
            })
            result_chunks.append(result_df)

        return pl.concat(result_chunks) if result_chunks else pl.DataFrame()

    def reassign_all_documents(self) -> None:
        assert self.ipca is not None and self.kmeans is not None, "Models must be initialized"
        logger.info("Reassigning all documents to current clusters (this may be slow)...")
        all_labels = []
        emb_files = sorted([f for f in os.listdir(os.path.join(self.output_dir, "embeddings")) if f.endswith(".npy")])
        for f in emb_files:
            path = os.path.join(self.output_dir, "embeddings", f)
            emb = load_embeddings(path)
            emb_reduced = self.ipca.transform(emb)
            labels = self.kmeans.predict(emb_reduced).tolist()
            all_labels.extend(labels)
        if len(all_labels) != len(self.docs):
            logger.warning("Reassigned labels count differs from docs count (%d vs %d). Skipping overwrite.", len(all_labels), len(self.docs))
        else:
            self.topic_assignments = [int(x) for x in all_labels]
            logger.info("Reassignment complete.")

    def build_topic_keywords(self, top_n: int = 12) -> dict[int, list[str]]:
        logger.info("Building topic keywords using c-TF-IDF style aggregation...")
        df = pl.DataFrame({
            "topic": self.topic_assignments,
            "doc": self.docs
        })
        grouped = df.group_by("topic").agg(pl.col("doc").str.concat(" ")).sort("topic")
        topics = grouped["topic"].to_list()
        docs_by_topic = grouped["doc"].to_list()

        vectorizer = CountVectorizer(stop_words="english", max_features=50000)
        term_matrix = vectorizer.fit_transform(docs_by_topic)
        tf = term_matrix.toarray().astype(float)
        df_counts = np.count_nonzero(tf > 0, axis=0)
        n_topics = len(topics)
        idf = np.log((n_topics + 1) / (df_counts + 1))
        ctfidf = tf * idf[np.newaxis, :]

        terms = np.array(vectorizer.get_feature_names_out())
        topic_keywords: dict[int, list[str]] = {}
        for i, t in enumerate(topics):
            row = ctfidf[i]
            top_idx = np.argsort(row)[::-1][:top_n]
            keywords = terms[top_idx].tolist()
            topic_keywords[t] = keywords

        return topic_keywords

    def build_topic_timeseries(self, time_freq: str = "D") -> pl.DataFrame:
        freq_map = {"H": "1h", "D": "1d", "W": "1w", "M": "1mo"}
        interval = freq_map.get(time_freq, "1d")

        df = pl.DataFrame({
            "topic": self.topic_assignments,
            "timestamp": self.timestamps
        })
        df = df.filter(pl.col("timestamp").is_not_null())

        df = df.with_columns(
            pl.col("timestamp").dt.truncate(interval).alias("time_bin")
        )

        ts = df.group_by(["topic", "time_bin"]).agg(pl.len().alias("count"))
        ts = ts.sort(["topic", "time_bin"])
        ts_pivot = ts.pivot(index="topic", columns="time_bin", values="count", aggregate_function="first")
        ts_pivot = ts_pivot.fill_null(0)

        return ts_pivot

    def plot_top_k_topics(self, ts: pl.DataFrame, topic_keywords: dict[int, list[str]], top_k: int = 10, time_freq: str = "D") -> go.Figure | None:
        if ts.shape[0] == 0:
            logger.warning("Empty timeseries — nothing to plot.")
            return None

        time_cols = [col for col in ts.columns if col != "topic"]
        if not time_cols:
            logger.warning("No time columns found in timeseries.")
            return None

        latest_col = time_cols[-1]
        topic_totals = ts.select([pl.col("topic"), pl.col(latest_col)])
        top_topics = topic_totals.sort(latest_col, descending=True).head(top_k)["topic"].to_list()

        fig = go.Figure()

        for topic in top_topics:
            topic_data = ts.filter(pl.col("topic") == topic)
            if topic_data.shape[0] > 0:
                values = [topic_data[col][0] if col in topic_data.columns else 0 for col in time_cols]
                keywords = topic_keywords.get(topic, [])[:3]
                label = f"Topic {topic}: {', '.join(keywords)}"

                fig.add_trace(go.Scatter(
                    x=time_cols,
                    y=values,
                    mode='lines+markers',
                    name=label,
                    hovertemplate=f'{label}<br>Time: %{{x}}<br>Count: %{{y}}<extra></extra>'
                ))

        fig.update_layout(
            title=f"Top {top_k} Topics Over Time ({time_freq})",
            xaxis_title="Time",
            yaxis_title="Document Count",
            hovermode='x unified',
            template='plotly_white',
            height=600,
            width=1200,
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=1.01)
        )

        return fig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["init_fit", "update", "assign"])
    p.add_argument("--input", "-i")
    p.add_argument("--output_dir", "-o")
    p.add_argument("--chunk_size", type=int, default=20000)
    p.add_argument("--pca_dim", type=int, default=64)
    p.add_argument("--n_clusters", type=int, default=100)
    p.add_argument("--embedding_model", type=str, default="all-MiniLM-L6-v2")
    p.add_argument("--batch_size_embed", type=int, default=256)
    p.add_argument("--time_freq", type=str, default="H")
    p.add_argument("--max_init_chunks", type=int, default=None)
    p.add_argument("--assign_only", action="store_true")
    return p.parse_args()


def load_config() -> Any:
    from experiments.configs import config

    class Args:
        mode = config.mode
        input = config.input_path
        output_dir = config.output_dir
        chunk_size = config.chunk_size
        pca_dim = config.pca_dim
        n_clusters = config.n_clusters
        embedding_model = config.embedding_model
        batch_size_embed = config.batch_size_embed
        time_freq = config.time_freq
        max_init_chunks = config.max_init_chunks
        assign_only = config.assign_only

    return Args()


def main(args: Any) -> None:
    import tempfile

    root = get_repo_root()
    input_path = resolve_path(args.input)
    output_dir = resolve_path(args.output_dir)

    ensure_dir(output_dir)

    tracking_db = root / "mlflow.db"
    artifact_root = (root / "mlruns").as_uri()
    ensure_dir(root / "mlruns")
    db_uri = f"sqlite:///{tracking_db}"
    mlflow.set_tracking_uri(db_uri)
    mlflow.set_registry_uri(db_uri)

    configure_experiment("streaming_bertopic", artifact_root)

    with mlflow.start_run(run_name=f"{args.mode}_{args.n_clusters}topics"):
        mlflow.log_params({
            "mode": args.mode,
            "embedding_model": args.embedding_model,
            "pca_dim": args.pca_dim,
            "n_clusters": args.n_clusters,
            "chunk_size": args.chunk_size,
            "batch_size_embed": args.batch_size_embed,
            "time_freq": args.time_freq,
            "max_init_chunks": args.max_init_chunks or "all"
        })

        pipeline = StreamingBERTopicPipeline(
            output_dir=str(output_dir),
            embedding_model_name=args.embedding_model,
            pca_dim=args.pca_dim,
            n_clusters=args.n_clusters,
            batch_size_embed=args.batch_size_embed
        )

        if args.mode == "init_fit":
            pipeline.initial_fit_stream(
                csv_path=str(input_path),
                text_col="tweet",
                id_col="id",
                created_at_col="created_at",
                chunk_size=args.chunk_size,
                max_chunks=args.max_init_chunks
            )
            kws = pipeline.build_topic_keywords()
            ts = pipeline.build_topic_timeseries(time_freq=args.time_freq)
            fig = pipeline.plot_top_k_topics(ts, kws, top_k=10, time_freq=args.time_freq)

            mlflow.log_metrics({
                "total_documents": len(pipeline.docs),
                "total_topics": len(set(pipeline.topic_assignments)),
                "unique_topics": len(kws)
            })

            with tempfile.TemporaryDirectory() as tmpdir:
                kw_path = os.path.join(tmpdir, "topic_keywords.json")
                with open(kw_path, "w", encoding="utf-8") as fh:
                    json.dump({str(k): v for k, v in kws.items()}, fh, indent=2)
                mlflow.log_artifact(kw_path)

                ts_path = os.path.join(tmpdir, f"topic_timeseries_{args.time_freq}.csv")
                ts.write_csv(ts_path)
                mlflow.log_artifact(ts_path)

                if fig is not None:
                    plot_path = os.path.join(tmpdir, f"top_10_topics_{args.time_freq}.html")
                    fig.write_html(plot_path)
                    mlflow.log_artifact(plot_path)

            if pipeline.ipca is not None:
                mlflow.sklearn.log_model(pipeline.ipca, name="ipca_model")
            if pipeline.kmeans is not None:
                mlflow.sklearn.log_model(pipeline.kmeans, name="kmeans_model")

        elif args.mode == "update":
            pipeline.load_state()
            pipeline.incremental_update(
                csv_path=str(input_path),
                text_col="tweet",
                id_col="id",
                created_at_col="created_at",
                chunk_size=args.chunk_size,
                update_keywords=False,
                reassign_on_update=False
            )
            kws = pipeline.build_topic_keywords()
            ts = pipeline.build_topic_timeseries(time_freq=args.time_freq)
            fig = pipeline.plot_top_k_topics(ts, kws, top_k=10, time_freq=args.time_freq)

            mlflow.log_metrics({
                "total_documents": len(pipeline.docs),
                "total_topics": len(set(pipeline.topic_assignments))
            })

            with tempfile.TemporaryDirectory() as tmpdir:
                kw_path = os.path.join(tmpdir, "topic_keywords.json")
                with open(kw_path, "w", encoding="utf-8") as fh:
                    json.dump({str(k): v for k, v in kws.items()}, fh, indent=2)
                mlflow.log_artifact(kw_path)

                ts_path = os.path.join(tmpdir, f"topic_timeseries_{args.time_freq}.csv")
                ts.write_csv(ts_path)
                mlflow.log_artifact(ts_path)

                if fig is not None:
                    plot_path = os.path.join(tmpdir, f"top_10_topics_{args.time_freq}.html")
                    fig.write_html(plot_path)
                    mlflow.log_artifact(plot_path)

        elif args.mode == "assign":
            pipeline.load_state()
            assigned_df = pipeline.assign_documents(
                csv_path=str(input_path),
                text_col="tweet",
                id_col="id",
                created_at_col="created_at",
                chunk_size=args.chunk_size
            )

            mlflow.log_metric("assigned_documents", assigned_df.shape[0])

            if args.assign_only:
                print(assigned_df.head(50).to_string())
            else:
                with tempfile.TemporaryDirectory() as tmpdir:
                    out_path = os.path.join(tmpdir, "assigned_docs.csv")
                    assigned_df.write_csv(out_path)
                    mlflow.log_artifact(out_path)
                logger.info("Logged assigned docs to MLflow")

        else:
            raise ValueError("Unknown mode: " + args.mode)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        args = parse_args()
    else:
        args = load_config()
    main(args)
