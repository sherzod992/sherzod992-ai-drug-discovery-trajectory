"""
Step 2: Topic Modeling with BERTopic
Analyzes patent abstracts to extract technology clusters and yearly trends.
Outputs: data/processed/topics_*.csv, topic_model_info.txt
"""
from __future__ import annotations

import os
import re
import json
import warnings
from pathlib import Path
from datetime import date

import pandas as pd
from tqdm import tqdm
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from hdbscan import HDBSCAN

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
OUTPUT = ROOT / "data" / "processed"
OUTPUT.mkdir(parents=True, exist_ok=True)

# Analysis settings
DATE_START = date(2016, 1, 1)
DATE_END = date(2025, 12, 31)

# Use the strict set first (higher quality AI+pharma overlap), fall back to broad
INPUT_FILES = [
    PROCESSED / "patents_preprocessed_strict.csv",
    PROCESSED / "patents_preprocessed.csv",
]

# Sample size for testing. Set to None to use all data.
# Recommended: 50000+ for stable topic patterns.
SAMPLE_SIZE = 50000


def load_data() -> pd.DataFrame:
    """Load the best available preprocessed patent dataset."""
    for f in INPUT_FILES:
        if f.exists():
            df = pd.read_csv(f, encoding="utf-8-sig")
            print(f"Loaded: {f.name} ({len(df):,} rows)")
            return df
    raise FileNotFoundError(
        f"No preprocessed CSV found in {PROCESSED}. "
        "Please run collect_stage1_data.py first."
    )


def clean_abstract(text: str) -> str:
    """Clean and filter English stop words from patent abstract text."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r"\s+", " ", text).strip()
    # Remove very short texts
    if len(text) < 50:
        return ""
    # Remove English stop words
    STOP_WORDS = {
        "the", "a", "an", "and", "or", "but", "of", "to", "in", "for",
        "on", "with", "at", "by", "from", "as", "is", "are", "was",
        "were", "be", "been", "being", "have", "has", "had", "do",
        "does", "did", "will", "would", "could", "should", "may",
        "might", "must", "can", "this", "that", "these", "those",
        "it", "its", "he", "she", "they", "them", "his", "her",
        "their", "we", "us", "you", "your", "i", "me", "my", "not",
        "no", "if", "then", "than", "so", "such", "only", "also",
        "just", "very", "too", "quite", "however", "therefore",
        "more", "most", "other", "some", "any", "all", "each",
        "every", "both", "few", "many", "much", "well", "now",
        "here", "there", "when", "where", "why", "how", "what",
        "which", "who", "whom", "whose", "into", "upon", "about",
        "between", "through", "during", "before", "after", "above",
        "below", "up", "down", "out", "off", "over", "under",
        "again", "further", "once", "same", "own", "s", "t",
    }
    words = [w for w in text.split() if w.lower() not in STOP_WORDS]
    return " ".join(words)


def filter_domain_relevance(df: pd.DataFrame) -> pd.DataFrame:
    """Filter out documents that are clearly not in the AI+pharma domain."""
    df = df.copy()
    # Count AI + drug keywords as a relevance score
    ai_keywords = [
        "artificial intelligence", "machine learning", "deep learning",
        "neural network", "generative ai", "large language model",
        "transformer", "bert", "gpt", "diffusion model", "gan",
        "computer vision", "natural language processing", "classifier",
        "drug discovery", "drug design", "pharmaceutical", "compound",
        "small molecule", "antibody", "biomarker", "clinical trial",
    ]
    df["abstract_lower"] = df["초록"].str.lower()
    df["relevance_score"] = df["abstract_lower"].apply(
        lambda t: sum(1 for kw in ai_keywords if kw in t)
    )
    # Keep documents with at least some AI or pharma signal
    df = df[df["relevance_score"] >= 1]
    df = df.drop(columns=["abstract_lower", "relevance_score"])
    return df


def main() -> None:
    # 1. Load data
    df = load_data()

    # 2. Prepare abstracts and dates
    df = df.copy()
    df["abstract_clean"] = df["초록"].apply(clean_abstract)
    df = df[df["abstract_clean"] != ""].copy()
    print(f"After cleaning abstracts: {len(df):,} rows")

    # 3. Filter domain relevance (AI + pharma)
    df = filter_domain_relevance(df)
    print(f"After domain filter: {len(df):,} rows")

    # Extract year from 출원일 (application date)
    df["출원일"] = pd.to_datetime(df["출원일"], errors="coerce")
    df = df.dropna(subset=["출원일"])
    df["연도"] = df["출원일"].dt.year
    df = df[(df["연도"] >= DATE_START.year) & (df["연도"] <= DATE_END.year)]
    print(f"After date filter ({DATE_START.year}-{DATE_END.year}): {len(df):,} rows")

    if len(df) < 100:
        raise ValueError("Too few documents for topic modeling. Need at least 100.")

    # Optional sampling for faster testing
    if SAMPLE_SIZE is not None and len(df) > SAMPLE_SIZE:
        df = df.sample(n=SAMPLE_SIZE, random_state=42).reset_index(drop=True)
        print(f"Sampled to {len(df):,} rows for testing")

    # Initialize BERTopic with stop words removed, conservative clustering
    print("\nInitializing BERTopic model...")
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    umap_model = PCA(
        n_components=5,
        random_state=42,
    )

    hdbscan_model = HDBSCAN(
        min_cluster_size=80,
        min_samples=10,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=False,
        core_dist_n_jobs=1,
    )

    topic_model = BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        language="english",
        calculate_probabilities=False,  # much faster on 50k docs
        verbose=True,
        nr_topics=12,
        top_n_words=15,
    )

    # 4. Fit model
    print(f"\nFitting BERTopic on {len(df):,} abstracts...")
    docs = df["abstract_clean"].tolist()
    topics, probs = topic_model.fit_transform(docs)

    # 5. Get topic info
    topic_info = topic_model.get_topic_info()
    print("\n=== Top Topics ===")
    print(topic_info.head(20).to_string(index=False))

    # Save topic info
    topic_info_path = OUTPUT / "topic_info.csv"
    topic_info.to_csv(topic_info_path, index=False, encoding="utf-8-sig")
    print(f"\nSaved topic info: {topic_info_path}")

    # 6. Assign topics to documents
    df["topic_id"] = topics
    df["topic_name"] = df["topic_id"].apply(
        lambda t: "_".join([w for w, _ in topic_model.get_topic(t)][:4]) if t != -1 else "outlier"
    )

    # Save document-topic mapping
    doc_topic_path = OUTPUT / "patent_topic_mapping.csv"
    df.to_csv(doc_topic_path, index=False, encoding="utf-8-sig")
    print(f"Saved patent-topic mapping: {doc_topic_path}")

    # 7. Yearly topic distribution (Hot/Cold topic analysis)
    yearly_topic = (
        df.groupby(["연도", "topic_id"])
        .size()
        .reset_index(name="count")
        .sort_values(["연도", "count"], ascending=[True, False])
    )

    yearly_path = OUTPUT / "yearly_topic_distribution.csv"
    yearly_topic.to_csv(yearly_path, index=False, encoding="utf-8-sig")
    print(f"Saved yearly topic distribution: {yearly_path}")

    # 8. Print top topics per year (Hot topics)
    print("\n=== Top 5 Topics per Year ===")
    for year in sorted(df["연도"].unique()):
        year_df = yearly_topic[yearly_topic["연도"] == year].head(5)
        print(f"\n--- {year}년 ---")
        for _, row in year_df.iterrows():
            tid = row["topic_id"]
            cnt = row["count"]
            if tid == -1:
                label = "outlier"
            else:
                words = [w for w, _ in topic_model.get_topic(tid)][:5]
                label = " + ".join(words)
            print(f"  Topic {tid} ({cnt}건): {label}")

    # 9. Topic size summary (excluding outliers)
    topic_sizes = topic_info[topic_info["Topic"] != -1].copy()
    topic_sizes = topic_sizes.sort_values("Count", ascending=False)
    print("\n=== Topic Sizes (excluding outliers) ===")
    print(topic_sizes.head(15).to_string(index=False))

    # 10. Save model summary
    n_topics = len(topic_info[topic_info["Topic"] != -1])
    n_outliers = (df["topic_id"] == -1).sum()
    summary = {
        "total_documents": len(df),
        "n_topics": int(n_topics),
        "n_outliers": int(n_outliers),
        "date_range": f"{DATE_START} .. {DATE_END}",
        "topics_per_year": {
            int(y): int(
                yearly_topic[yearly_topic["연도"] == y]["count"].sum()
            )
            for y in sorted(df["연도"].unique())
        },
    }
    summary_path = OUTPUT / "topic_model_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\nSaved model summary: {summary_path}")
    print(f"\n=== Summary ===")
    print(f"Documents: {len(df):,}")
    print(f"Topics: {n_topics}")
    print(f"Outliers: {n_outliers:,}")


if __name__ == "__main__":
    main()