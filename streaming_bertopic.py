#!/usr/bin/env python3
"""
streaming_bertopic.py

Incremental streaming topic pipeline:
- Incremental PCA for dimensionality reduction
- MiniBatchKMeans for online clustering
- SentenceTransformer embeddings computed per-batch and cached to disk
- c-TF-IDF style topic keywords
- Topic time-series (hourly/daily)
- Persisted pipeline state to resume

Usage (initial fit on CSV):
python streaming_bertopic.py --mode init_fit --input tweets.csv --output_dir out --chunk_size 20000 --pca_dim 64 --n_clusters 100 --time_freq H

Usage (incremental update from a new CSV of new tweets):
python streaming_bertopic.py --mode update --input new_tweets.csv --output_dir out --chunk_size 5000

Usage (assign new single-file to existing model without updating model):
python streaming_bertopic.py --mode assign --input new_tweets.csv --output_dir out --assign_only
"""

import os
import argparse
import logging
from datetime import datetime
from typing import List, Tuple, Dict, Optional
import json

import numpy as np
import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import IncrementalPCA
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
logging.getLogger().setLevel(logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------
# Utilities
# ---------------------------
def ensure_dir(d):
    os.makedirs(d, exist_ok=True)


def parse_timestamp(df, created_at_col="created_at", date_col="date", time_col="time"):
    if created_at_col in df.columns:
        ts = pd.to_datetime(df[created_at_col], errors="coerce")
    elif date_col in df.columns and time_col in df.columns:
        ts = pd.to_datetime(df[date_col].astype(str) + " " + df[time_col].astype(str), errors="coerce")
    else:
        ts = pd.to_datetime(df.iloc[:, 0], errors="coerce")  # fallback
    return ts


def clean_text(s: str) -> str:
    import re
    if not isinstance(s, str):
        return ""
    s = s.lower()
    s = re.sub(r"http\S+", " ", s)
    s = re.sub(r"@\w+", " ", s)
    s = re.sub(r"#", "", s)
    s = re.sub(r"[^\x00-\x7F]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


# ---------------------------
# Embeddings caching
# ---------------------------
def embeddings_cache_path(output_dir: str, model_name: str, chunk_idx: int):
    safe_model = model_name.replace("/", "_")
    return os.path.join(output_dir, "embeddings", f"emb_{safe_model}_chunk{chunk_idx:05d}.npy")


def save_embeddings(emb: np.ndarray, path: str):
    ensure_dir(os.path.dirname(path))
    np.save(path, emb)
    logger.info("Saved embeddings %s shape=%s", path, emb.shape)


def load_embeddings(path: str) -> np.ndarray:
    return np.load(path)


# ---------------------------
# Incremental pipeline class
# ---------------------------
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
        self.embedder = SentenceTransformer(self.embedding_model_name)
        self.pca_dim = pca_dim
        self.n_clusters = n_clusters
        self.batch_size_embed = batch_size_embed

        # Models (initialized during fit or loaded)
        self.ipca: Optional[IncrementalPCA] = None
        self.kmeans: Optional[MiniBatchKMeans] = None

        # Metadata
        self.doc_index = []  # list of doc ids (strings)
        self.docs = []       # cleaned text for each doc (same order)
        self.timestamps = [] # timestamp for each doc (pd.Timestamp)
        self.topic_assignments = []  # assigned cluster per doc

        # For persistence filepaths
        self.state_path = os.path.join(self.output_dir, "pipeline_state.joblib")
        self.meta_path = os.path.join(self.output_dir, "pipeline_meta.json")

    # ---------------------------
    # Persistence
    # ---------------------------
    def save_state(self):
        logger.info("Saving pipeline state...")
        state = {
            "embedding_model_name": self.embedding_model_name,
            "pca_dim": self.pca_dim,
            "n_clusters": self.n_clusters,
            "batch_size_embed": self.batch_size_embed,
            "doc_index": self.doc_index,
            "timestamps": [ts.isoformat() if not pd.isna(ts) else None for ts in self.timestamps],
            "topic_assignments": self.topic_assignments,
            "docs_len": len(self.docs)
        }
        # Save text docs separately (to avoid joblib large memory)
        docs_path = os.path.join(self.output_dir, "docs.txt")
        with open(docs_path, "w", encoding="utf-8") as fh:
            for d in self.docs:
                fh.write(d.replace("\n", " ") + "\n")

        with open(self.meta_path, "w", encoding="utf-8") as fh:
            json.dump(state, fh)

        # Save models using joblib
        model_objs = {"ipca": self.ipca, "kmeans": self.kmeans}
        joblib.dump(model_objs, self.state_path)
        logger.info("Saved pipeline metadata to %s and models to %s", self.meta_path, self.state_path)

    def load_state(self):
        if not os.path.exists(self.meta_path) or not os.path.exists(self.state_path):
            raise FileNotFoundError("Pipeline state not found in output_dir.")
        logger.info("Loading pipeline state from %s ...", self.output_dir)
        with open(self.meta_path, "r", encoding="utf-8") as fh:
            state = json.load(fh)
        self.embedding_model_name = state["embedding_model_name"]
        self.embedder = SentenceTransformer(self.embedding_model_name)
        self.pca_dim = state["pca_dim"]
        self.n_clusters = state["n_clusters"]
        self.batch_size_embed = state["batch_size_embed"] if "batch_size_embed" in state else self.batch_size_embed

        # load docs
        docs_path = os.path.join(self.output_dir, "docs.txt")
        self.docs = []
        with open(docs_path, "r", encoding="utf-8") as fh:
            for line in fh:
                self.docs.append(line.strip())

        self.doc_index = state["doc_index"]
        self.timestamps = [pd.to_datetime(x) if x is not None else pd.NaT for x in state["timestamps"]]
        self.topic_assignments = state["topic_assignments"]

        models = joblib.load(self.state_path)
        self.ipca = models.get("ipca")
        self.kmeans = models.get("kmeans")
        logger.info("Loaded pipeline: %d docs, %d topics", len(self.docs), len(set(self.topic_assignments)))

    # ---------------------------
    # Helpers
    # ---------------------------
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        return self.embedder.encode(texts, batch_size=self.batch_size_embed, show_progress_bar=False, convert_to_numpy=True)

    # ---------------------------
    # Initial incremental fit over CSV chunk-by-chunk
    # ---------------------------
    def initial_fit_stream(self,
                           csv_path: str,
                           text_col: str = "tweet",
                           id_col: str = "id",
                           created_at_col: str = "created_at",
                           chunk_size: int = 20000,
                           max_chunks: Optional[int] = None):
        """
        Stream CSV in chunks, compute embeddings per-chunk and update IncrementalPCA and MiniBatchKMeans
        Save per-chunk embeddings to disk for reuse.
        """
        logger.info("Starting initial fit stream from %s", csv_path)

        # Initialize incremental models
        self.ipca = IncrementalPCA(n_components=self.pca_dim)
        self.kmeans = MiniBatchKMeans(n_clusters=self.n_clusters, batch_size=1024)

        chunk_idx = 0
        reader = pd.read_csv(csv_path, dtype=str, chunksize=chunk_size)

        for chunk in reader:
            chunk_idx += 1
            if max_chunks is not None and chunk_idx > max_chunks:
                break

            logger.info("Processing chunk %d", chunk_idx)
            # basic cleaning & parse timestamp
            chunk = chunk.dropna(subset=[text_col])
            chunk["cleaned"] = chunk[text_col].apply(clean_text)
            chunk["timestamp"] = parse_timestamp(chunk, created_at_col=created_at_col)

            docs = chunk["cleaned"].tolist()
            ids = chunk[id_col].tolist()
            ts_list = chunk["timestamp"].tolist()

            # Embedding compute
            batch_emb = self.embed_texts(docs)  # (n, dim)
            save_path = embeddings_cache_path(self.output_dir, self.embedding_model_name, chunk_idx)
            save_embeddings(batch_emb, save_path)

            # PCA partial fit (IPCA expects 2D arrays)
            try:
                self.ipca.partial_fit(batch_emb)
            except Exception as e:
                logger.warning("IPCA partial_fit failed: %s", e)
                # fallback: if first call and shapes cause error, fit normally on small sample
                self.ipca.fit(batch_emb)

            # Transform to low-dim
            emb_reduced = self.ipca.transform(batch_emb)

            # KMeans partial fit
            try:
                self.kmeans.partial_fit(emb_reduced)
            except Exception as e:
                logger.warning("KMeans partial_fit failed: %s", e)
                # fallback: try fit if first chunk
                self.kmeans.fit(emb_reduced)

            # Predict cluster assignments
            labels = self.kmeans.predict(emb_reduced).tolist()

            # Append metadata to pipeline in-memory stores
            start_idx = len(self.docs)
            for i, doc in enumerate(docs):
                self.doc_index.append(ids[i])
                self.docs.append(doc)
                self.timestamps.append(ts_list[i])
                self.topic_assignments.append(int(labels[i]))

            logger.info("Chunk %d processed: %d docs appended (total %d)", chunk_idx, len(docs), len(self.docs))

        # After streaming, we may want to save state
        self.save_state()
        logger.info("Initial streaming fit complete. Total docs: %d", len(self.docs))

        # Build topic keywords after initial fit
        self.build_topic_keywords(save=True)

    # ---------------------------
    # Update pipeline with new CSV (incremental)
    # ---------------------------
    def incremental_update(self,
                           csv_path: str,
                           text_col: str = "tweet",
                           id_col: str = "id",
                           created_at_col: str = "created_at",
                           chunk_size: int = 5000,
                           update_keywords: bool = True,
                           reassign_on_update: bool = False):
        """
        Ingest new CSV of new tweets and update IPCA and KMeans incrementally.
        If reassign_on_update True, reassign all docs to nearest cluster (costly).
        """
        if self.ipca is None or self.kmeans is None:
            raise RuntimeError("Pipeline models not initialized. Call initial_fit_stream or load_state first.")

        reader = pd.read_csv(csv_path, dtype=str, chunksize=chunk_size)
        added = 0
        # find starting chunk_id for caching embeddings
        existing_chunks = len([f for f in os.listdir(os.path.join(self.output_dir, "embeddings")) if f.endswith(".npy")]) \
                          if os.path.exists(os.path.join(self.output_dir, "embeddings")) else 0
        chunk_idx = existing_chunks

        for chunk in reader:
            chunk_idx += 1
            chunk = chunk.dropna(subset=[text_col])
            chunk["cleaned"] = chunk[text_col].apply(clean_text)
            chunk["timestamp"] = parse_timestamp(chunk, created_at_col=created_at_col)

            docs = chunk["cleaned"].tolist()
            ids = chunk[id_col].tolist()
            ts_list = chunk["timestamp"].tolist()

            batch_emb = self.embed_texts(docs)
            save_path = embeddings_cache_path(self.output_dir, self.embedding_model_name, chunk_idx)
            save_embeddings(batch_emb, save_path)

            # incremental PCA update
            try:
                self.ipca.partial_fit(batch_emb)
            except Exception as e:
                logger.warning("IPCA partial_fit failed on update: %s", e)

            emb_reduced = self.ipca.transform(batch_emb)

            # update kmeans
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
        # Optionally reassign all docs (costly). Usually unnecessary; new docs assigned to clusters directly.
        if reassign_on_update:
            self.reassign_all_documents()

        if update_keywords:
            self.build_topic_keywords(save=True)
        self.save_state()

    # ---------------------------
    # Assign new docs to existing model (no model update)
    # ---------------------------
    def assign_documents(self,
                         csv_path: str,
                         text_col: str = "tweet",
                         id_col: str = "id",
                         created_at_col: str = "created_at",
                         chunk_size: int = 2000) -> pd.DataFrame:
        """
        Assign docs in csv_path to current topics using existing IPCA+KMeans (no model updates).
        Returns a dataframe of assigned rows with topic.
        """
        if self.ipca is None or self.kmeans is None:
            raise RuntimeError("Pipeline models not initialized. Call load_state first.")

        out_rows = []
        reader = pd.read_csv(csv_path, dtype=str, chunksize=chunk_size)
        for chunk in reader:
            chunk = chunk.dropna(subset=[text_col])
            chunk["cleaned"] = chunk[text_col].apply(clean_text)
            chunk["timestamp"] = parse_timestamp(chunk, created_at_col=created_at_col)

            docs = chunk["cleaned"].tolist()
            ids = chunk[id_col].tolist()
            ts_list = chunk["timestamp"].tolist()

            batch_emb = self.embed_texts(docs)
            emb_reduced = self.ipca.transform(batch_emb)
            labels = self.kmeans.predict(emb_reduced).tolist()

            for i in range(len(docs)):
                out_rows.append({
                    id_col: ids[i],
                    "timestamp": ts_list[i],
                    "cleaned_text": docs[i],
                    "topic": int(labels[i])
                })
        return pd.DataFrame(out_rows)

    # ---------------------------
    # Optional: reassign all docs to nearest clusters (costly)
    # ---------------------------
    def reassign_all_documents(self):
        logger.info("Reassigning all documents to current clusters (this may be slow)...")
        # load embeddings from cache sequentially and predict
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

    # ---------------------------
    # Topic keywords via c-TF-IDF style
    # ---------------------------
    def build_topic_keywords(self, save: bool = True, top_n: int = 12):
        logger.info("Building topic keywords using c-TF-IDF style aggregation...")
        df = pd.DataFrame({
            "topic": self.topic_assignments,
            "doc": self.docs
        })
        grouped = df.groupby("topic")["doc"].apply(lambda docs: " ".join(docs)).to_dict()
        topics = sorted(grouped.keys())

        vectorizer = CountVectorizer(stop_words="english", max_features=50000)
        docs_by_topic = [grouped[t] for t in topics]
        X = vectorizer.fit_transform(docs_by_topic)  # shape: (n_topics, vocab)
        # compute c-TF-IDF: tf * log(N / df)
        tf = X.toarray().astype(float)
        df_counts = np.count_nonzero(tf > 0, axis=0)
        n_topics = len(topics)
        idf = np.log((n_topics + 1) / (df_counts + 1))  # smoothed idf
        ctfidf = tf * idf[np.newaxis, :]

        terms = np.array(vectorizer.get_feature_names_out())
        topic_keywords = {}
        for i, t in enumerate(topics):
            row = ctfidf[i]
            top_idx = np.argsort(row)[::-1][:top_n]
            keywords = terms[top_idx].tolist()
            topic_keywords[t] = keywords

        # Save keywords to JSON
        if save:
            kw_path = os.path.join(self.output_dir, "topic_keywords.json")
            with open(kw_path, "w", encoding="utf-8") as fh:
                json.dump({str(k): v for k, v in topic_keywords.items()}, fh, indent=2)
            logger.info("Saved topic keywords to %s", kw_path)
        return topic_keywords

    # ---------------------------
    # Topic timeseries & plotting
    # ---------------------------
    def build_topic_timeseries(self, time_freq: str = "D") -> pd.DataFrame:
        # Build DataFrame of topic counts per period
        df = pd.DataFrame({
            "topic": self.topic_assignments,
            "timestamp": pd.to_datetime(self.timestamps)
        })
        df = df.dropna(subset=["timestamp"])
        df["time_bin"] = df["timestamp"].dt.to_period(time_freq).dt.to_timestamp()
        ts = df.groupby(["topic", "time_bin"]).size().unstack(fill_value=0)
        ts = ts.sort_index(axis=1)
        ts_path = os.path.join(self.output_dir, f"topic_timeseries_{time_freq}.csv")
        ts.to_csv(ts_path)
        logger.info("Saved topic timeseries to %s", ts_path)
        return ts

    def plot_top_k_topics(self, ts: pd.DataFrame, topic_keywords: Dict[int, List[str]], top_k: int = 10, time_freq: str = "D"):
        if ts.shape[0] == 0:
            logger.warning("Empty timeseries — nothing to plot.")
            return
        latest = ts.iloc[:, -1]
        top_topics = latest.sort_values(ascending=False).head(top_k).index.tolist()
        plt.figure(figsize=(14, 7))
        for t in top_topics:
            label = f"{t}:" + (", ".join(topic_keywords.get(t, [])[:3]))
            plt.plot(ts.columns, ts.loc[t], label=label)
        plt.xlabel("Time")
        plt.ylabel("Count")
        plt.title(f"Top {top_k} topics over time ({time_freq})")
        plt.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0)
        out = os.path.join(self.output_dir, f"top_{top_k}_topics_{time_freq}.png")
        plt.tight_layout()
        plt.savefig(out, dpi=150)
        plt.close()
        logger.info("Saved plot to %s", out)


# ---------------------------
# CLI / main
# ---------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Streaming BERTopic pipeline")
    p.add_argument("--mode", choices=["init_fit", "update", "assign"], required=True,
                   help="init_fit: initial streaming fit from CSV; update: incremental update; assign: assign new docs to existing model without updating")
    p.add_argument("--input", "-i", required=True, help="Input CSV path")
    p.add_argument("--output_dir", "-o", required=True, help="Output directory for pipeline state")
    p.add_argument("--chunk_size", type=int, default=20000, help="CSV chunk size for streaming operations")
    p.add_argument("--pca_dim", type=int, default=64, help="Incremental PCA output dimension")
    p.add_argument("--n_clusters", type=int, default=100, help="Number of clusters for MiniBatchKMeans")
    p.add_argument("--embedding_model", type=str, default="all-MiniLM-L6-v2", help="SentenceTransformer model name")
    p.add_argument("--batch_size_embed", type=int, default=256, help="Embedding batch size")
    p.add_argument("--time_freq", type=str, default="H", help="Time frequency for monitoring: H hourly, D daily")
    p.add_argument("--max_init_chunks", type=int, default=None, help="Limit number of chunks during initial fit (for testing)")
    p.add_argument("--assign_only", action="store_true", help="If set in assign mode: do not save; just print assignments")
    return p.parse_args()


def main(args):
    ensure_dir(args.output_dir)

    pipeline = StreamingBERTopicPipeline(
        output_dir=args.output_dir,
        embedding_model_name=args.embedding_model,
        pca_dim=args.pca_dim,
        n_clusters=args.n_clusters,
        batch_size_embed=args.batch_size_embed
    )

    if args.mode == "init_fit":
        # run initial streaming fit
        pipeline.initial_fit_stream(
            csv_path=args.input,
            text_col="tweet",
            id_col="id",
            created_at_col="created_at",
            chunk_size=args.chunk_size,
            max_chunks=args.max_init_chunks
        )
        # after fit, build timeseries and plot
        kws = pipeline.build_topic_keywords(save=True)
        ts = pipeline.build_topic_timeseries(time_freq=args.time_freq)
        pipeline.plot_top_k_topics(ts, kws, top_k=10, time_freq=args.time_freq)

    elif args.mode == "update":
        # load existing pipeline state
        pipeline.load_state()
        pipeline.incremental_update(
            csv_path=args.input,
            text_col="tweet",
            id_col="id",
            created_at_col="created_at",
            chunk_size=args.chunk_size,
            update_keywords=True,
            reassign_on_update=False
        )
        kws = pipeline.build_topic_keywords(save=True)
        ts = pipeline.build_topic_timeseries(time_freq=args.time_freq)
        pipeline.plot_top_k_topics(ts, kws, top_k=10, time_freq=args.time_freq)

    elif args.mode == "assign":
        # assign new docs to existing model without updating it
        pipeline.load_state()
        assigned_df = pipeline.assign_documents(
            csv_path=args.input,
            text_col="tweet",
            id_col="id",
            created_at_col="created_at",
            chunk_size=args.chunk_size
        )
        if args.assign_only:
            print(assigned_df.head(50).to_string())
        else:
            out_path = os.path.join(args.output_dir, "assigned_new_docs.csv")
            assigned_df.to_csv(out_path, index=False)
            logger.info("Saved assigned docs to %s", out_path)

    else:
        raise ValueError("Unknown mode: " + args.mode)


if __name__ == "__main__":
    args = parse_args()
    main(args)
