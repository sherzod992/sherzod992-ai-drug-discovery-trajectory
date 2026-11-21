"""Phase 4: Assignee comparison + RTA on v2 data (display assignees excluded)."""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "data" / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

INPUT = PROCESSED / "patents_preprocessed_v2.csv"
TOPIC_MAP = PROCESSED / "patent_topic_mapping_v2.csv"
FOCAL = ("G16H", "G16C", "G06N", "G06V", "A61K", "A61B", "A61M", "C07K", "C07D", "G16Z")

BIG_TECH = (
    "INTERNATIONAL BUSINESS MACHINES", "IBM ", "GOOGLE", "MICROSOFT", "AMAZON",
    "APPLE INC", "META PLATFORMS", "FACEBOOK", "NVIDIA", "INTEL CORPORATION",
    "QUALCOMM", "ORACLE", "SAP SE", "ADOBE", "DEEPMIND", "BAIDU", "TENCENT",
    "ALIBABA", "SAMSUNG ELECTRONICS", "LG ELECTRONICS", "NEC CORPORATION",
    "HUAWEI", "PALANTIR", "SALESFORCE", "CISCO", "DELL ", "HEWLETT",
    "SIEMENS", "PHILIPS", "ACCENTURE",
)
BIG_PHARMA = (
    "NOVARTIS", "PFIZER", "MERCK", "ROCHE", "GENENTECH", "BRISTOL-MYERS",
    "BRISTOL MYERS", "SANOFI", "ASTRAZENECA", "GLAXOSMITHKLINE", "GSK ",
    "JOHNSON & JOHNSON", "JANSSEN", "ELI LILLY", "LILLY AND COMPANY", "ABBVIE",
    "AMGEN", "GILEAD", "BIOGEN", "REGENERON", "TAKEDA", "BAYER", "BOEHRINGER",
    "NOVO NORDISK", "MODERNA", "BIONTECH", "VERTEX", "INCYTE", "HOFFMANN-LA ROCHE",
)
ACADEMIA = (
    "UNIVERSITY", "INSTITUTE OF TECHNOLOGY", "COLLEGE", "SCHOOL OF MEDICINE",
    "REGENTS OF", "BOARD OF TRUSTEES", "MASSACHUSETTS INSTITUTE",
    "DEPARTMENT OF HEALTH AND HUMAN SERVICES",
)
DISPLAY_BLOCK = (
    "SAMSUNG DISPLAY", "LG DISPLAY", "BOE TECHNOLOGY", "AU OPTRONICS",
    "JAPAN DISPLAY", "UNIVERSAL DISPLAY", "E INK",
)


def normalize(name: str) -> str:
    n = (name or "").upper().strip()
    n = re.sub(r"\s+", " ", n)
    return n


def classify(name: str) -> str:
    n = normalize(name)
    if not n:
        return "unknown"
    if any(d in n for d in DISPLAY_BLOCK):
        return "other"  # exclude from tech contrast
    if any(x in n for x in BIG_PHARMA):
        return "big_pharma"
    if any(x in n for x in BIG_TECH):
        return "big_tech"
    if any(x in n for x in ACADEMIA):
        return "academia"
    return "other"


def parse_codes(cell: str) -> list[str]:
    if not isinstance(cell, str):
        return []
    return [p.strip().upper() for p in re.split(r"[;|]", cell) if p.strip()]


def subclass(code: str) -> str:
    return code[:4] if code else ""


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def main() -> None:
    df = pd.read_csv(INPUT, encoding="utf-8-sig")
    df["assignee_norm"] = df["출원인"].fillna("").map(normalize)
    # drop display-only assignees entirely from analysis contrast set
    display_mask = df["assignee_norm"].apply(lambda n: any(d in n for d in DISPLAY_BLOCK))
    n_disp = int(display_mask.sum())
    df = df[~display_mask].copy()
    df["sector"] = df["assignee_norm"].map(classify)
    df["출원일"] = pd.to_datetime(df["출원일"], errors="coerce")
    df["연도"] = df["출원일"].dt.year
    print(f"Patents after display drop ({n_disp}): {len(df):,}")

    # CPC shares by sector
    cpc_rows = []
    for sector, sub in df.groupby("sector"):
        total = len(sub)
        counts = Counter()
        for cell in sub["CPC분류코드"].fillna(""):
            scs = {subclass(c) for c in parse_codes(cell)}
            for s in scs:
                if s in FOCAL:
                    counts[s] += 1
        for s, n in counts.items():
            cpc_rows.append(
                {
                    "sector": sector,
                    "subclass": s,
                    "patent_count": n,
                    "share": round(n / total, 4) if total else 0,
                    "sector_total": total,
                }
            )
    cpc_port = pd.DataFrame(cpc_rows)

    # Topic join
    topic_port = pd.DataFrame()
    if TOPIC_MAP.exists():
        tm = pd.read_csv(TOPIC_MAP, encoding="utf-8-sig")
        key = "특허번호"
        merged = df.merge(tm[[key, "topic_id", "topic_name"]], on=key, how="inner")
        print(f"Topic join: {len(merged):,}")
        topic_port = (
            merged.groupby(["sector", "topic_id", "topic_name"], as_index=False)
            .size()
            .rename(columns={"size": "patent_count"})
        )
        sector_n = merged.groupby("sector")[key].nunique().to_dict()
        topic_port["share"] = topic_port.apply(
            lambda r: round(r["patent_count"] / sector_n.get(r["sector"], 1), 4), axis=1
        )

        # RTA for big_tech / big_pharma on topics (excl outliers)
        labeled = merged[merged["topic_id"] != -1]
        N = len(labeled)
        Nj = labeled.groupby("topic_id").size().to_dict()
        rta_rows = []
        for sector in ("big_tech", "big_pharma"):
            sub = labeled[labeled["sector"] == sector]
            ni = len(sub)
            nij = sub.groupby("topic_id").size().to_dict()
            for tid, n_ij in nij.items():
                rta = (n_ij / ni) / (Nj[tid] / N) if ni and Nj.get(tid) else 0
                rta_rows.append(
                    {
                        "sector": sector,
                        "topic_id": int(tid),
                        "n_ij": int(n_ij),
                        "n_i": int(ni),
                        "N_j": int(Nj[tid]),
                        "N": int(N),
                        "RTA": round(float(rta), 4),
                    }
                )
        rta_df = pd.DataFrame(rta_rows)
        rta_df.to_csv(PROCESSED / "assignee_rta_v2.csv", index=False, encoding="utf-8-sig")
        rta_df.to_csv(RESULTS / "assignee_rta_v2.csv", index=False, encoding="utf-8-sig")

        # cosine similarity of topic distributions
        topics = sorted(labeled["topic_id"].unique())
        vecs = {}
        for sector in ("big_tech", "big_pharma", "academia"):
            sub = labeled[labeled["sector"] == sector]
            cnt = sub.groupby("topic_id").size()
            v = np.array([cnt.get(t, 0) for t in topics], dtype=float)
            vecs[sector] = v / v.sum() if v.sum() else v
        cos = {
            "big_tech_vs_big_pharma": round(cosine(vecs["big_tech"], vecs["big_pharma"]), 4),
            "big_tech_vs_academia": round(cosine(vecs["big_tech"], vecs["academia"]), 4),
            "big_pharma_vs_academia": round(cosine(vecs["big_pharma"], vecs["academia"]), 4),
        }
    else:
        rta_df = pd.DataFrame()
        cos = {}

    top = (
        df.groupby(["assignee_norm", "sector"], as_index=False)
        .size()
        .rename(columns={"size": "patent_count"})
        .sort_values("patent_count", ascending=False)
    )
    sector_totals = df.groupby("sector", as_index=False).size().rename(columns={"size": "patent_count"})
    yearly = df.groupby(["연도", "sector"], as_index=False).size().rename(columns={"size": "patent_count"})

    top.to_csv(PROCESSED / "assignee_top_v2.csv", index=False, encoding="utf-8-sig")
    top.to_csv(RESULTS / "assignee_top_v2.csv", index=False, encoding="utf-8-sig")
    cpc_port.to_csv(PROCESSED / "assignee_sector_cpc_portfolio_v2.csv", index=False, encoding="utf-8-sig")
    cpc_port.to_csv(RESULTS / "assignee_sector_cpc_portfolio_v2.csv", index=False, encoding="utf-8-sig")
    yearly.to_csv(PROCESSED / "assignee_sector_yearly_v2.csv", index=False, encoding="utf-8-sig")
    yearly.to_csv(RESULTS / "assignee_sector_yearly_v2.csv", index=False, encoding="utf-8-sig")
    if not topic_port.empty:
        topic_port.to_csv(PROCESSED / "assignee_sector_topic_portfolio_v2.csv", index=False, encoding="utf-8-sig")
        topic_port.to_csv(RESULTS / "assignee_sector_topic_portfolio_v2.csv", index=False, encoding="utf-8-sig")

    def pivot(sector: str) -> dict:
        sub = cpc_port[cpc_port["sector"] == sector]
        return {r["subclass"]: r["share"] for _, r in sub.iterrows()}

    summary = {
        "version": "v2",
        "n_patents": int(len(df)),
        "display_assignees_removed": n_disp,
        "sector_totals": sector_totals.to_dict(orient="records"),
        "top_big_tech": top[top["sector"] == "big_tech"].head(10).to_dict(orient="records"),
        "top_big_pharma": top[top["sector"] == "big_pharma"].head(10).to_dict(orient="records"),
        "cpc_contrast": {
            "big_tech": pivot("big_tech"),
            "big_pharma": pivot("big_pharma"),
        },
        "topic_cosine_similarity": cos,
        "rta_preview": rta_df.head(20).to_dict(orient="records") if not rta_df.empty else [],
    }
    (PROCESSED / "assignee_comparison_summary_v2.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (RESULTS / "assignee_comparison_summary_v2.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2)[:2500])


if __name__ == "__main__":
    main()
