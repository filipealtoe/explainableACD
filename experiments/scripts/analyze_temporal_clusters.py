"""
Temporal Distribution Analysis via Clustering
Verify if Train/Dev/Test splits come from different time periods
by checking if they form distinct clusters in embedding space.

Usage:
    python experiments/scripts/analyze_temporal_clusters.py
    python experiments/scripts/analyze_temporal_clusters.py --n-clusters 20
    python experiments/scripts/analyze_temporal_clusters.py --fast  # Skip visualization
"""

import argparse
from pathlib import Path

import numpy as np
import polars as pl
from scipy.stats import chi2_contingency
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from umap import UMAP


def load_splits() -> pl.DataFrame:
    """Load train/dev/test and combine with split labels."""
    base = Path("data/raw/CT24_checkworthy_english")

    train = pl.read_csv(base / "CT24_checkworthy_english_train.tsv", separator="\t")
    dev = pl.read_csv(base / "CT24_checkworthy_english_dev.tsv", separator="\t")
    test = pl.read_csv(base / "CT24_checkworthy_english_test_gold.tsv", separator="\t")

    train = train.with_columns(pl.lit("train").alias("split"))
    dev = dev.with_columns(pl.lit("dev").alias("split"))
    test = test.with_columns(pl.lit("test").alias("split"))

    return pl.concat([train, dev, test])


def embed_sentences(texts: list[str], model_name: str = "BAAI/bge-large-en-v1.5") -> np.ndarray:
    """Embed sentences using SentenceTransformer."""
    print(f"Loading model: {model_name}")
    model = SentenceTransformer(model_name)
    print(f"Embedding {len(texts)} sentences...")
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=64)
    return embeddings


def cluster_embeddings(
    embeddings: np.ndarray, n_clusters: int = 15, umap_dim: int = 50
) -> tuple[np.ndarray, np.ndarray]:
    """Reduce dimensions with UMAP and cluster with K-means."""
    print(f"UMAP reduction to {umap_dim} dimensions...")
    reducer = UMAP(n_components=umap_dim, metric="cosine", random_state=42, verbose=True)
    reduced = reducer.fit_transform(embeddings)

    print(f"K-means clustering with {n_clusters} clusters...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(reduced)

    print("UMAP reduction to 2D for visualization...")
    reducer_2d = UMAP(n_components=2, metric="cosine", random_state=42, verbose=False)
    coords_2d = reducer_2d.fit_transform(embeddings)

    return labels, coords_2d


def analyze_clusters(df: pl.DataFrame, cluster_labels: np.ndarray) -> pl.DataFrame:
    """Analyze split distribution within each cluster."""
    df = df.with_columns(pl.Series("cluster", cluster_labels))

    # Calculate expected test percentage (baseline)
    total_test = len(df.filter(pl.col("split") == "test"))
    total_all = len(df)
    expected_test_pct = total_test / total_all * 100

    print("\n" + "=" * 75)
    print("CLUSTER COMPOSITION BY SPLIT")
    print(f"(Expected test %: {expected_test_pct:.2f}% if evenly distributed)")
    print("=" * 75)

    results = []
    for cluster_id in sorted(df["cluster"].unique()):
        cluster_df = df.filter(pl.col("cluster") == cluster_id)
        total = len(cluster_df)

        train_count = len(cluster_df.filter(pl.col("split") == "train"))
        dev_count = len(cluster_df.filter(pl.col("split") == "dev"))
        test_count = len(cluster_df.filter(pl.col("split") == "test"))

        train_pct = train_count / total * 100
        dev_pct = dev_count / total * 100
        test_pct = test_count / total * 100

        # Flag clusters where test is overrepresented (>3x expected)
        flag = ""
        if test_pct > expected_test_pct * 3:
            flag = " ⚠️  TEST-HEAVY"
        elif test_pct == 0 and total > 100:
            flag = " 📜 NO TEST (old era?)"

        results.append(
            {
                "cluster": cluster_id,
                "total": total,
                "train_pct": train_pct,
                "dev_pct": dev_pct,
                "test_pct": test_pct,
                "flag": flag,
            }
        )

        print(f"\nCluster {cluster_id} (n={total}):{flag}")
        print(f"  Train: {train_count:>5} ({train_pct:>5.1f}%)")
        print(f"  Dev:   {dev_count:>5} ({dev_pct:>5.1f}%)")
        print(f"  Test:  {test_count:>5} ({test_pct:>5.1f}%)")

        # Sample sentences from this cluster
        sample = cluster_df.sample(min(2, total), seed=42)["Text"].to_list()
        for i, s in enumerate(sample):
            print(f"  Sample {i + 1}: {s[:100]}...")

    # Summary stats
    test_heavy = [r for r in results if "TEST-HEAVY" in r["flag"]]
    no_test = [r for r in results if "NO TEST" in r["flag"]]

    print("\n" + "=" * 75)
    print("SUMMARY")
    print("=" * 75)
    print(f"Total clusters: {len(results)}")
    print(f"TEST-HEAVY clusters (>3x expected test %): {len(test_heavy)}")
    print(f"NO-TEST clusters (0% test, n>100): {len(no_test)}")

    if test_heavy:
        test_in_heavy = sum(
            len(df.filter((pl.col("cluster") == r["cluster"]) & (pl.col("split") == "test")))
            for r in test_heavy
        )
        print(f"Test sentences in TEST-HEAVY clusters: {test_in_heavy}/{total_test} ({test_in_heavy / total_test * 100:.1f}%)")

    return df


def extract_cluster_topics(
    df: pl.DataFrame, cluster_labels: np.ndarray, n_top_words: int = 10
) -> dict[int, dict]:
    """Extract representative topics/keywords for each cluster using c-TF-IDF."""
    from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

    df = df.with_columns(pl.Series("cluster", cluster_labels))
    texts = df["Text"].to_list()

    # Get total counts per split for normalization
    total_train = len(df.filter(pl.col("split") == "train"))
    total_dev = len(df.filter(pl.col("split") == "dev"))
    total_test = len(df.filter(pl.col("split") == "test"))

    # Create document per cluster (concatenate all sentences)
    cluster_docs = {}
    for cluster_id in sorted(df["cluster"].unique()):
        cluster_texts = df.filter(pl.col("cluster") == cluster_id)["Text"].to_list()
        cluster_docs[cluster_id] = " ".join(cluster_texts)

    # Use TF-IDF to find distinctive words per cluster
    vectorizer = TfidfVectorizer(
        max_features=5000,
        stop_words="english",
        ngram_range=(1, 2),  # Unigrams and bigrams
        min_df=2,
        max_df=0.8,
    )

    # Fit on all cluster documents
    cluster_ids = sorted(cluster_docs.keys())
    docs = [cluster_docs[cid] for cid in cluster_ids]
    tfidf_matrix = vectorizer.fit_transform(docs)
    feature_names = vectorizer.get_feature_names_out()

    # Extract top words per cluster
    cluster_topics = {}
    for idx, cluster_id in enumerate(cluster_ids):
        # Get TF-IDF scores for this cluster
        scores = tfidf_matrix[idx].toarray().flatten()
        top_indices = scores.argsort()[-n_top_words:][::-1]
        top_words = [(feature_names[i], scores[i]) for i in top_indices]

        # Also get sample sentences
        cluster_df = df.filter(pl.col("cluster") == cluster_id)
        samples = cluster_df.sample(min(3, len(cluster_df)), seed=42)["Text"].to_list()

        # Get counts per split in this cluster
        train_in_cluster = len(cluster_df.filter(pl.col("split") == "train"))
        dev_in_cluster = len(cluster_df.filter(pl.col("split") == "dev"))
        test_in_cluster = len(cluster_df.filter(pl.col("split") == "test"))
        total = len(cluster_df)

        # RAW percentages (within cluster) - what % of this cluster is train/dev/test
        train_pct_raw = train_in_cluster / total * 100 if total > 0 else 0
        dev_pct_raw = dev_in_cluster / total * 100 if total > 0 else 0
        test_pct_raw = test_in_cluster / total * 100 if total > 0 else 0

        # NORMALIZED percentages - what % of each SPLIT is in this cluster
        # This answers: "Of all test sentences, what % are in this cluster?"
        train_pct_norm = train_in_cluster / total_train * 100 if total_train > 0 else 0
        dev_pct_norm = dev_in_cluster / total_dev * 100 if total_dev > 0 else 0
        test_pct_norm = test_in_cluster / total_test * 100 if total_test > 0 else 0

        cluster_topics[cluster_id] = {
            "top_words": top_words,
            "samples": samples,
            "total": total,
            # Raw (within cluster)
            "train_pct_raw": train_pct_raw,
            "dev_pct_raw": dev_pct_raw,
            "test_pct_raw": test_pct_raw,
            # Normalized (of each split)
            "train_pct_norm": train_pct_norm,
            "dev_pct_norm": dev_pct_norm,
            "test_pct_norm": test_pct_norm,
            # Counts
            "train_count": train_in_cluster,
            "dev_count": dev_in_cluster,
            "test_count": test_in_cluster,
        }

    return cluster_topics


def print_cluster_topics(cluster_topics: dict[int, dict]) -> None:
    """Print cluster topics in a readable format."""
    print("\n" + "=" * 85)
    print("CLUSTER SEMANTIC ANALYSIS: Topics and Representative Keywords")
    print("=" * 85)
    print("NORMALIZED %: What % of each SPLIT is in this cluster")
    print("  (e.g., 'Test 25%' means 25% of all test sentences are in this cluster)")
    print("=" * 85)

    # Sort clusters by normalized test percentage (descending) to show test-heavy first
    sorted_clusters = sorted(cluster_topics.items(), key=lambda x: -x[1]["test_pct_norm"])

    for cluster_id, info in sorted_clusters:
        # Determine cluster type based on normalized percentages
        # A cluster is TEST-HEAVY if it has disproportionate test representation
        # (test_norm much higher than train_norm)
        test_ratio = info["test_pct_norm"] / max(info["train_pct_norm"], 0.01)

        flag = ""
        if info["test_pct_norm"] > 5 and test_ratio > 2:
            flag = "⚠️  TEST-HEAVY (test overrepresented)"
        elif info["test_pct_norm"] == 0 and info["total"] > 50:
            flag = "📜 NO-TEST (likely old era)"

        print(f"\n{'─' * 85}")
        print(f"CLUSTER {cluster_id} (n={info['total']}) {flag}")
        print(f"  Normalized %: Train {info['train_pct_norm']:.1f}% | Dev {info['dev_pct_norm']:.1f}% | Test {info['test_pct_norm']:.1f}%")
        print(f"  Raw counts:   Train {info['train_count']:,} | Dev {info['dev_count']:,} | Test {info['test_count']:,}")
        print(f"{'─' * 85}")

        # Top keywords
        keywords = [word for word, score in info["top_words"][:8]]
        print(f"  📌 Top Keywords: {', '.join(keywords)}")

        # Infer semantic meaning from keywords
        topic_hints = infer_topic_meaning(keywords, info)
        if topic_hints:
            print(f"  🎯 Likely Topic: {topic_hints}")

        # Sample sentences
        print(f"  📝 Sample Sentences:")
        for i, sample in enumerate(info["samples"][:2], 1):
            truncated = sample[:120] + "..." if len(sample) > 120 else sample
            print(f"     {i}. \"{truncated}\"")


def infer_topic_meaning(keywords: list[str], info: dict) -> str:
    """Infer semantic meaning from keywords and split distribution."""
    keywords_lower = [k.lower() for k in keywords]
    keywords_str = " ".join(keywords_lower)

    # Topic patterns
    patterns = {
        # Foreign policy / Military
        "military|troops|war|iraq|afghanistan|weapons|nuclear|iran|korea|terrorism|terrorist": "Foreign Policy / Military",
        "soviet|russia|cold war|kremlin|communist": "Cold War Era Politics",
        "china|trade|tariff|beijing|xi": "China Relations / Trade",

        # Economy
        "economy|jobs|unemployment|tax|taxes|budget|deficit|debt|spending": "Economy / Fiscal Policy",
        "trade|nafta|tpp|import|export|manufacturing": "Trade Policy",
        "wall street|banks|financial|bailout|recession": "Financial Crisis / Banking",

        # Healthcare
        "health|healthcare|insurance|medicare|medicaid|obamacare|affordable": "Healthcare Policy",
        "drug|pharmaceutical|prescription": "Pharmaceutical / Drug Policy",

        # Social issues
        "immigration|immigrant|border|illegal|asylum|daca": "Immigration Policy",
        "climate|environment|energy|oil|gas|carbon|renewable": "Climate / Energy Policy",
        "gun|weapons|assault|nra|background check": "Gun Policy",
        "abortion|women|reproductive": "Reproductive Rights",
        "education|school|college|student|teacher": "Education Policy",

        # Politics / Process
        "vote|voter|election|ballot|campaign": "Elections / Voting",
        "congress|senate|house|legislation|bill": "Congressional / Legislative",
        "president|administration|white house|oval": "Presidential / Executive",
        "supreme court|justice|constitutional": "Judicial / Constitutional",

        # Trump era specific
        "trump|donald|maga": "Trump Era Politics",
        "impeach|ukraine|mueller|russia investigation": "Trump Investigations",

        # Debate meta
        "senator|governor|congressman|secretary": "Political Titles / Debate Format",
        "agree|disagree|point|question|answer": "Debate Rhetoric / Meta",
    }

    import re
    matches = []
    for pattern, topic in patterns.items():
        if re.search(pattern, keywords_str):
            matches.append(topic)

    # Add era hint based on NORMALIZED split distribution
    era_hint = ""
    test_ratio = info["test_pct_norm"] / max(info["train_pct_norm"], 0.01)
    if info["test_pct_norm"] > 5 and test_ratio > 2:
        era_hint = " [2019-2020 Dem Primary]"
    elif info["test_pct_norm"] == 0 and info["total"] > 100:
        era_hint = " [Pre-2016 Debates]"

    if matches:
        return ", ".join(matches[:2]) + era_hint
    return era_hint.strip() if era_hint else "General Political Discourse"


def save_cluster_topics(cluster_topics: dict[int, dict], save_path: str) -> None:
    """Save cluster topics to a text file."""
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    with open(save_path, "w") as f:
        f.write("CLUSTER SEMANTIC ANALYSIS: Topics and Representative Keywords\n")
        f.write("=" * 85 + "\n")
        f.write(f"Generated with {len(cluster_topics)} clusters\n")
        f.write("NORMALIZED %: What % of each SPLIT is in this cluster\n")
        f.write("  (e.g., 'Test 25%' means 25% of all test sentences are in this cluster)\n")
        f.write("=" * 85 + "\n\n")

        # Sort clusters by normalized test percentage (descending)
        sorted_clusters = sorted(cluster_topics.items(), key=lambda x: -x[1]["test_pct_norm"])

        for cluster_id, info in sorted_clusters:
            test_ratio = info["test_pct_norm"] / max(info["train_pct_norm"], 0.01)

            flag = ""
            if info["test_pct_norm"] > 5 and test_ratio > 2:
                flag = "⚠️  TEST-HEAVY (test overrepresented)"
            elif info["test_pct_norm"] == 0 and info["total"] > 50:
                flag = "📜 NO-TEST (likely old era)"

            f.write(f"{'─' * 85}\n")
            f.write(f"CLUSTER {cluster_id} (n={info['total']}) {flag}\n")
            f.write(f"  Normalized %: Train {info['train_pct_norm']:.1f}% | Dev {info['dev_pct_norm']:.1f}% | Test {info['test_pct_norm']:.1f}%\n")
            f.write(f"  Raw counts:   Train {info['train_count']:,} | Dev {info['dev_count']:,} | Test {info['test_count']:,}\n")
            f.write(f"{'─' * 85}\n")

            keywords = [word for word, score in info["top_words"][:8]]
            f.write(f"  Top Keywords: {', '.join(keywords)}\n")

            topic_hints = infer_topic_meaning(keywords, info)
            if topic_hints:
                f.write(f"  Likely Topic: {topic_hints}\n")

            f.write(f"  Sample Sentences:\n")
            for i, sample in enumerate(info["samples"][:2], 1):
                truncated = sample[:120] + "..." if len(sample) > 120 else sample
                f.write(f"     {i}. \"{truncated}\"\n")
            f.write("\n")

    print(f"📋 Saved cluster topics to: {save_path}")


def test_cluster_independence(df: pl.DataFrame) -> tuple[float, float]:
    """Chi-square test: Are splits distributed differently across clusters?"""
    contingency = (
        df.group_by(["cluster", "split"])
        .agg(pl.count().alias("count"))
        .pivot(on="split", index="cluster", values="count")
        .fill_null(0)
    )

    # Ensure columns exist and are in correct order
    for col in ["train", "dev", "test"]:
        if col not in contingency.columns:
            contingency = contingency.with_columns(pl.lit(0).alias(col))

    matrix = contingency.select(["train", "dev", "test"]).to_numpy()

    chi2, p_value, dof, expected = chi2_contingency(matrix)

    print("\n" + "=" * 75)
    print("CHI-SQUARE TEST: Are splits independent of clusters?")
    print("=" * 75)
    print(f"Chi-square statistic: {chi2:.2f}")
    print(f"P-value: {p_value:.2e}")
    print(f"Degrees of freedom: {dof}")

    if p_value < 0.001:
        print("\n✅ RESULT: Splits are NOT independent of clusters (p < 0.001)")
        print("   → Train/Dev/Test come from DIFFERENT distributions")
        print("   → This confirms temporal/domain shift between splits")
    elif p_value < 0.05:
        print("\n⚠️  RESULT: Weak evidence of dependence (p < 0.05)")
        print("   → Some difference between splits, but not overwhelming")
    else:
        print("\n❌ RESULT: Cannot reject independence (p >= 0.05)")
        print("   → No strong evidence that splits differ systematically")

    return chi2, p_value


def plot_clusters(
    coords_2d: np.ndarray,
    splits: list[str],
    cluster_labels: np.ndarray,
    save_path: str = "experiments/results/temporal_cluster_analysis.png",
) -> None:
    """Generate UMAP visualization colored by split and cluster with annotations."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    import matplotlib.colors as mcolors

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    # Plot 1: Color by split
    split_colors = {"train": "#3498db", "dev": "#f39c12", "test": "#e74c3c"}

    # Plot test last so it's visible on top
    split_order = ["train", "dev", "test"]
    for split in split_order:
        mask = [s == split for s in splits]
        mask_coords = coords_2d[mask]
        axes[0].scatter(
            mask_coords[:, 0],
            mask_coords[:, 1],
            c=split_colors[split],
            alpha=0.4 if split != "test" else 0.8,
            s=3 if split != "test" else 15,
            label=f"{split} (n={sum(mask)})",
        )

    axes[0].set_title("UMAP: Colored by Split\n(Do Test sentences cluster separately?)", fontsize=12)
    axes[0].set_xlabel("UMAP 1")
    axes[0].set_ylabel("UMAP 2")
    axes[0].legend(loc="upper right")

    # Plot 2: Color by cluster with NORMALIZED percentage annotations
    n_clusters = len(set(cluster_labels))
    cmap = plt.cm.get_cmap("tab20", n_clusters)

    scatter = axes[1].scatter(
        coords_2d[:, 0], coords_2d[:, 1], c=cluster_labels, cmap="tab20", alpha=0.4, s=3
    )

    # Calculate totals for normalization
    splits_arr = np.array(splits)
    total_train = (splits_arr == "train").sum()
    total_dev = (splits_arr == "dev").sum()
    total_test = (splits_arr == "test").sum()

    # Calculate cluster centroids and NORMALIZED split percentages
    cluster_stats = []

    for cluster_id in range(n_clusters):
        mask = cluster_labels == cluster_id
        if mask.sum() == 0:
            continue

        # Centroid
        centroid_x = coords_2d[mask, 0].mean()
        centroid_y = coords_2d[mask, 1].mean()

        # Counts per split in this cluster
        cluster_splits = splits_arr[mask]
        total = len(cluster_splits)
        train_in_cluster = (cluster_splits == "train").sum()
        dev_in_cluster = (cluster_splits == "dev").sum()
        test_in_cluster = (cluster_splits == "test").sum()

        # NORMALIZED percentages: what % of each SPLIT is in this cluster
        train_pct_norm = train_in_cluster / total_train * 100 if total_train > 0 else 0
        dev_pct_norm = dev_in_cluster / total_dev * 100 if total_dev > 0 else 0
        test_pct_norm = test_in_cluster / total_test * 100 if total_test > 0 else 0

        cluster_stats.append({
            "id": cluster_id,
            "centroid": (centroid_x, centroid_y),
            "train_pct_norm": train_pct_norm,
            "dev_pct_norm": dev_pct_norm,
            "test_pct_norm": test_pct_norm,
            "total": total,
        })

        # Annotate cluster with NORMALIZED percentages
        # Color-code based on whether test is overrepresented
        test_ratio = test_pct_norm / max(train_pct_norm, 0.01)

        if test_pct_norm > 5 and test_ratio > 2:
            text_color = "#e74c3c"  # Red for test-heavy
            fontweight = "bold"
        elif test_pct_norm == 0 and total > 50:
            text_color = "#3498db"  # Blue for no-test
            fontweight = "normal"
        else:
            text_color = "black"
            fontweight = "normal"

        # Show all three splits with normalized percentages
        annotation = f"C{cluster_id} (n={total})\nTr:{train_pct_norm:.1f}% Dv:{dev_pct_norm:.1f}%\nTe:{test_pct_norm:.1f}%"
        axes[1].annotate(
            annotation,
            (centroid_x, centroid_y),
            fontsize=7,
            ha="center",
            va="center",
            color=text_color,
            fontweight=fontweight,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7, edgecolor="gray"),
        )

    axes[1].set_title("UMAP: Colored by Cluster (NORMALIZED %)\n(% of each split in cluster | Red=Test overrepresented, Blue=No-test)", fontsize=11)
    axes[1].set_xlabel("UMAP 1")
    axes[1].set_ylabel("UMAP 2")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\n📊 Saved visualization to: {save_path}")

    # Also save a detailed breakdown table with NORMALIZED percentages
    table_path = save_path.replace(".png", "_breakdown.txt")
    with open(table_path, "w") as f:
        f.write("CLUSTER SPLIT BREAKDOWN (NORMALIZED %)\n")
        f.write("=" * 75 + "\n")
        f.write("NORMALIZED %: What % of each SPLIT is in this cluster\n")
        f.write("  (e.g., 'Test 25%' means 25% of all test sentences are in this cluster)\n")
        f.write("=" * 75 + "\n")
        f.write(f"{'Cluster':<8} {'Total':<8} {'Train%':<10} {'Dev%':<10} {'Test%':<10} {'Flag':<20}\n")
        f.write("-" * 75 + "\n")

        for stats in sorted(cluster_stats, key=lambda x: -x["test_pct_norm"]):
            test_ratio = stats["test_pct_norm"] / max(stats["train_pct_norm"], 0.01)
            flag = ""
            if stats["test_pct_norm"] > 5 and test_ratio > 2:
                flag = "⚠️ TEST OVERREPRESENTED"
            elif stats["test_pct_norm"] == 0 and stats["total"] > 50:
                flag = "📜 NO-TEST"

            f.write(f"C{stats['id']:<7} {stats['total']:<8} {stats['train_pct_norm']:<10.1f} {stats['dev_pct_norm']:<10.1f} {stats['test_pct_norm']:<10.1f} {flag}\n")

        f.write("-" * 75 + "\n")

    print(f"📋 Saved breakdown table to: {table_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze temporal distribution via clustering")
    parser.add_argument("--n-clusters", type=int, default=15, help="Number of clusters (default: 15)")
    parser.add_argument("--model", type=str, default="BAAI/bge-large-en-v1.5", help="Sentence transformer model")
    parser.add_argument("--fast", action="store_true", help="Skip visualization (faster)")
    parser.add_argument("--save-embeddings", type=str, help="Save embeddings to this path")
    parser.add_argument("--load-embeddings", type=str, help="Load pre-computed embeddings")
    args = parser.parse_args()

    print("=" * 75)
    print("TEMPORAL DISTRIBUTION ANALYSIS VIA CLUSTERING")
    print("=" * 75)

    # Load data
    print("\n📂 Loading data...")
    df = load_splits()
    print(f"Total sentences: {len(df):,}")
    print(f"  Train: {len(df.filter(pl.col('split') == 'train')):,}")
    print(f"  Dev:   {len(df.filter(pl.col('split') == 'dev')):,}")
    print(f"  Test:  {len(df.filter(pl.col('split') == 'test')):,}")

    # Embed or load embeddings
    texts = df["Text"].to_list()

    if args.load_embeddings:
        print(f"\n📥 Loading embeddings from {args.load_embeddings}")
        embeddings = np.load(args.load_embeddings)
    else:
        print(f"\n🔢 Embedding sentences with {args.model}...")
        embeddings = embed_sentences(texts, args.model)

        if args.save_embeddings:
            print(f"💾 Saving embeddings to {args.save_embeddings}")
            np.save(args.save_embeddings, embeddings)

    # Cluster
    print(f"\n🔮 Clustering into {args.n_clusters} clusters...")
    cluster_labels, coords_2d = cluster_embeddings(embeddings, n_clusters=args.n_clusters)

    # Analyze
    print("\n📊 Analyzing cluster composition...")
    df = analyze_clusters(df, cluster_labels)

    # Extract and print cluster topics
    print("\n🏷️  Extracting cluster topics...")
    cluster_topics = extract_cluster_topics(df, cluster_labels)
    print_cluster_topics(cluster_topics)

    # Save topics to file
    topics_path = "experiments/results/temporal_cluster_topics.txt"
    save_cluster_topics(cluster_topics, topics_path)

    # Statistical test
    chi2, p_value = test_cluster_independence(df)

    # Visualize
    if not args.fast:
        print("\n🎨 Generating visualization...")
        plot_clusters(coords_2d, df["split"].to_list(), cluster_labels)
    else:
        print("\n⏭️  Skipping visualization (--fast mode)")

    print("\n" + "=" * 75)
    print("DONE")
    print("=" * 75)


if __name__ == "__main__":
    main()
