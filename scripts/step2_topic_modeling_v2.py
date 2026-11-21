"""
Phase 2: SciBERT embeddings (cached) + UMAP + HDBSCAN BERTopic on v2 set.
"""
from __future__ import annotations

import json
import re
import warnings
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from bertopic import BERTopic
from hdbscan import HDBSCAN
from sentence_transformers import SentenceTransformer, models
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "data" / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

DATE_START = date(2016, 1, 1)
DATE_END = date(2025, 12, 31)
INPUT = PROCESSED / "patents_preprocessed_v2.csv"
EMB_CACHE = PROCESSED / "embeddings_scibert_v2.npy"
RANDOM_STATE = 42

STOP_WORDS = {
    "the", "a", "an", "and", "or", "but", "of", "to", "in", "for", "on", "with",
    "at", "by", "from", "as", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could", "should",
    "may", "might", "must", "can", "this", "that", "these", "those", "it", "its",
    "method", "methods", "comprising", "thereof", "invention", "present", "said",
    "wherein", "disclosed", "herein", "embodiment", "embodiments",
}


def clean_abstract(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) < 50:
        return ""
    tokens = [w for w in re.findall(r"[A-Za-z][A-Za-z\-]+", text.lower()) if w not in STOP_WORDS]
    return " ".join(tokens)


def build_embedding_model():
    try:
        word = models.Transformer("allenai/scibert_scivocab_uncased", max_seq_length=256)
        pool = models.Pooling(word.get_word_embedding_dimension(), pooling_mode_mean_tokens=True)
        model = SentenceTransformer(modules=[word, pool])
        model.encode(["test"], show_progress_bar=False)
        return model, "allenai/scibert_scivocab_uncased"
    except Exception as e:
        print(f"SciBERT failed ({e}); using S-PubMedBert")
        name = "pritamdeka/S-PubMedBert-MS-MARCO"
        return SentenceTransformer(name), name


def main() -> None:
    df = pd.read_csv(INPUT, encoding="utf-8-sig")
    df["출원일"] = pd.to_datetime(df["출원일"], errors="coerce")
    df = df.dropna(subset=["출원일", "초록"])
    df["연도"] = df["출원일"].dt.year
    df = df[(df["연도"] >= DATE_START.year) & (df["연도"] <= DATE_END.year)]
    df["abstract_clean"] = df["초록"].apply(clean_abstract)
    df = df[df["abstract_clean"].str.len() > 0].reset_index(drop=True)
    docs = df["abstract_clean"].tolist()
    print(f"Docs: {len(docs):,}")

    embedding_model, emb_name = build_embedding_model()
    print(f"Model: {emb_name}")

    if EMB_CACHE.exists() and EMB_CACHE.stat().st_size > 1000:
        embeddings = np.load(EMB_CACHE)
        if len(embeddings) != len(docs):
            print("Cache size mismatch; re-encoding")
            embeddings = None
        else:
            print(f"Loaded embeddings cache: {embeddings.shape}")
    else:
        embeddings = None

    if embeddings is None:
        print("Encoding with SciBERT (once)...")
        embeddings = embedding_model.encode(docs, show_progress_bar=True, batch_size=32)
        np.save(EMB_CACHE, embeddings)
        print(f"Cached {EMB_CACHE}")

    # HDBSCAN often collapses on this dense SciBERT space; use KMeans for stable multi-topic split
    n_clusters = 8
    print(f"\n=== KMeans n_clusters={n_clusters} (SciBERT + PCA) ===")
    umap_model = PCA(n_components=50, random_state=RANDOM_STATE)
    cluster_model = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init=10)
    topic_model = BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=cluster_model,
        calculate_probabilities=False,
        verbose=False,
        top_n_words=12,
    )
    topics, _ = topic_model.fit_transform(docs, embeddings=embeddings)
    out_rate = float(np.mean(np.array(topics) == -1))
    n_top = len(set(topics) - {-1})
    mcs, ms = n_clusters, 0
    print(f"outlier={out_rate:.1%} topics={n_top}")
    if n_top < 4:
        raise RuntimeError(f"Expected >=4 topics, got {n_top}")
    print(f"\nSelected KMeans k={n_clusters} outlier={out_rate:.1%} topics={n_top}")

    topic_info = topic_model.get_topic_info()
    df = df.copy()
    df["topic_id"] = topics
    df["topic_name"] = [
        ("_".join([w for w, _ in topic_model.get_topic(t)][:4]) if t != -1 else "outlier")
        for t in topics
    ]

    yearly = df.groupby(["연도", "topic_id"]).size().reset_index(name="count")
    totals = yearly.groupby("연도")["count"].transform("sum")
    yearly["share"] = yearly["count"] / totals
    slopes = {}
    for tid in sorted(set(topics)):
        sub = yearly[yearly["topic_id"] == tid].set_index("연도").reindex(range(2016, 2026)).fillna(0)
        y = sub["share"].values.astype(float)
        slopes[int(tid)] = float(np.polyfit(np.arange(len(y)), y, 1)[0]) if y.sum() > 0 else 0.0

    topic_info.to_csv(PROCESSED / "topic_info_v2.csv", index=False, encoding="utf-8-sig")
    topic_info.to_csv(RESULTS / "topic_info_v2.csv", index=False, encoding="utf-8-sig")
    df.to_csv(PROCESSED / "patent_topic_mapping_v2.csv", index=False, encoding="utf-8-sig")
    yearly.to_csv(PROCESSED / "yearly_topic_distribution_v2.csv", index=False, encoding="utf-8-sig")
    yearly.to_csv(RESULTS / "yearly_topic_distribution_v2.csv", index=False, encoding="utf-8-sig")

    summary = {
        "version": "v2",
        "embedding": emb_name,
        "cluster_method": "KMeans",
        "n_clusters": mcs,
        "min_samples": ms,
        "total_documents": int(len(df)),
        "n_topics": int(n_top),
        "n_outliers": int((df["topic_id"] == -1).sum()),
        "outlier_rate": round(out_rate, 4),
        "date_range": f"{DATE_START} .. {DATE_END}",
        "topic_share_slopes": slopes,
    }
    (PROCESSED / "topic_model_summary_v2.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (RESULTS / "topic_model_summary_v2.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(topic_info.head(20).to_string(index=False))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
