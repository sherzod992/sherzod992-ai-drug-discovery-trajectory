"""
Step 4: Assignee portfolio comparison (Big Pharma vs Big Tech).
Inputs:  data/processed/patents_preprocessed.csv
         data/processed/patent_topic_mapping.csv (optional)
Outputs: data/processed/assignee_*.csv|json
"""
from __future__ import annotations

import json
import re
from collections import Counter
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
OUT = PROCESSED
OUT.mkdir(parents=True, exist_ok=True)

DATE_START = date(2016, 1, 1)
DATE_END = date(2025, 12, 31)

FOCAL_SUBCLASSES = ("G16H", "G16C", "G06N", "G06V", "A61K", "A61B", "A61M", "C07K", "C07D", "G16Z")

# Name fragments (uppercase) for coarse sector labels
BIG_TECH = (
    "INTERNATIONAL BUSINESS MACHINES",
    "IBM ",
    "GOOGLE",
    "MICROSOFT",
    "AMAZON",
    "APPLE INC",
    "META PLATFORMS",
    "FACEBOOK",
    "NVIDIA",
    "INTEL CORPORATION",
    "QUALCOMM",
    "ORACLE",
    "SAP SE",
    "ADOBE",
    "DEEPMIND",
    "BAIDU",
    "TENCENT",
    "ALIBABA",
    "SAMSUNG ELECTRONICS",
    "LG ELECTRONICS",
    "NEC CORPORATION",
    "HUAWEI",
    "SNAP INC",
    "TWITTER",
    "UBER ",
    "PALANTIR",
    "SALESFORCE",
    "CISCO",
    "DELL ",
    "HP INC",
    "HEWLETT",
    "SONY ",
    "PANASONIC",
    "SIEMENS",  # tech/medtech conglomerate — counted as tech for portfolio contrast
    "PHILIPS",
    "ACCENTURE",
    "CAPITAL ONE",
    "BANK OF AMERICA",
    "INTUIT",
)

BIG_PHARMA = (
    "NOVARTIS",
    "PFIZER",
    "MERCK",
    "ROCHE",
    "GENENTECH",
    "BRISTOL-MYERS",
    "BRISTOL MYERS",
    "SANOFI",
    "ASTRAZENECA",
    "GLAXOSMITHKLINE",
    "GSK ",
    "JOHNSON & JOHNSON",
    "JANSSEN",
    "ELI LILLY",
    "LILLY AND COMPANY",
    "ABBVIE",
    "AMGEN",
    "GILEAD",
    "BIOGEN",
    "REGENERON",
    "TAKEDA",
    "BAYER",
    "BOEHRINGER",
    "NOVO NORDISK",
    "TEVA ",
    "ASTELLAS",
    "DAIICHI",
    "EISAI",
    "MODERNA",
    "BIONTECH",
    "VERTEX PHARM",
    "INCYTE",
    "ILLUMINA",  # life-science tools, kept under pharma/life-sci bucket
    "THERMO FISHER",
)

ACADEMIA = (
    "UNIVERSITY",
    "UNIVERSITE",
    "COLLEGE",
    "INSTITUTE OF TECHNOLOGY",
    "MASSACHUSETTS INSTITUTE",
    "BOARD OF REGENTS",
    "BOARD OF TRUSTEES",
    "THE REGENTS",
    "HARVARD",
    "STANFORD",
    "MAX-PLANCK",
    "CENTRE NATIONAL",
    "CNRS",
    "RIKEN",
    "FRAUNHOFER",
    "DEPARTMENT OF HEALTH AND HUMAN SERVICES",
    "UNITED STATES OF AMERICA",
    "GOVERNMENT",
)


def normalize_name(name: str) -> str:
    if not isinstance(name, str):
        return ""
    n = re.sub(r"\s+", " ", name).strip().upper()
    return n


def classify_assignee(name: str) -> str:
    n = normalize_name(name)
    if not n:
        return "unknown"
    # Samsung Display is devices; still tech-adjacent
    if "SAMSUNG DISPLAY" in n or "SAMSUNG SDI" in n:
        return "big_tech"
    for frag in BIG_PHARMA:
        if frag in n:
            return "big_pharma"
    for frag in BIG_TECH:
        if frag in n:
            return "big_tech"
    for frag in ACADEMIA:
        if frag in n:
            return "academia"
    return "other"


def subclass_of(code: str) -> str:
    c = (code or "").strip().upper().replace(" ", "")
    return c[:4] if c else ""


def parse_subclasses(cell: str) -> list[str]:
    if not isinstance(cell, str) or not cell.strip():
        return []
    out = []
    seen = set()
    for p in cell.replace("|", ";").split(";"):
        s = subclass_of(p.strip())
        if s and s not in seen:
            seen.add(s)
            out.append(s)
    return out


def main() -> None:
    path = PROCESSED / "patents_preprocessed.csv"
    df = pd.read_csv(path, encoding="utf-8-sig")
    print(f"Loaded {path.name}: {len(df):,} rows")

    df["출원일"] = pd.to_datetime(df["출원일"], errors="coerce")
    df = df.dropna(subset=["출원일"])
    df["연도"] = df["출원일"].dt.year
    df = df[(df["연도"] >= DATE_START.year) & (df["연도"] <= DATE_END.year)].copy()

    df["assignee_norm"] = df["출원인"].apply(normalize_name)
    df["sector"] = df["assignee_norm"].apply(classify_assignee)
    df["subclasses"] = df["CPC분류코드"].apply(parse_subclasses)

    # --- Top assignees ---
    top = (
        df[df["assignee_norm"] != ""]
        .groupby(["assignee_norm", "sector"], as_index=False)
        .size()
        .rename(columns={"size": "patent_count"})
        .sort_values("patent_count", ascending=False)
    )
    top_path = OUT / "assignee_top.csv"
    top.to_csv(top_path, index=False, encoding="utf-8-sig")
    print(f"Saved {top_path} (top={len(top):,})")

    # --- Sector yearly ---
    sector_yearly = (
        df.groupby(["연도", "sector"], as_index=False)
        .size()
        .rename(columns={"size": "patent_count"})
        .sort_values(["연도", "patent_count"], ascending=[True, False])
    )
    sector_yearly_path = OUT / "assignee_sector_yearly.csv"
    sector_yearly.to_csv(sector_yearly_path, index=False, encoding="utf-8-sig")

    sector_totals = (
        df.groupby("sector", as_index=False)
        .size()
        .rename(columns={"size": "patent_count"})
        .sort_values("patent_count", ascending=False)
    )

    # --- CPC portfolio by sector (focal subclasses) ---
    rows = []
    for sector, sub in df.groupby("sector"):
        cnt = Counter()
        for codes in sub["subclasses"]:
            for c in set(codes):
                if c in FOCAL_SUBCLASSES:
                    cnt[c] += 1
        total = len(sub)
        for c, n in cnt.items():
            rows.append(
                {
                    "sector": sector,
                    "subclass": c,
                    "patent_count": n,
                    "share": round(n / total, 4) if total else 0.0,
                    "sector_total": total,
                }
            )
    cpc_port = pd.DataFrame(rows).sort_values(
        ["sector", "patent_count"], ascending=[True, False]
    )
    cpc_port_path = OUT / "assignee_sector_cpc_portfolio.csv"
    cpc_port.to_csv(cpc_port_path, index=False, encoding="utf-8-sig")

    # --- Topic portfolio by sector (if mapping available) ---
    topic_port = pd.DataFrame()
    topic_path = PROCESSED / "patent_topic_mapping.csv"
    if topic_path.exists():
        tm = pd.read_csv(topic_path, encoding="utf-8-sig")
        # join keys
        id_col = "특허번호" if "특허번호" in tm.columns else "patent_id"
        tm[id_col] = tm[id_col].astype(str)
        df["특허번호"] = df["특허번호"].astype(str)
        merged = df.merge(
            tm[[id_col, "topic_id", "topic_name"]].drop_duplicates(id_col),
            left_on="특허번호",
            right_on=id_col,
            how="inner",
        )
        print(f"Topic join: {len(merged):,} patents with topic labels")
        if len(merged):
            topic_port = (
                merged.groupby(["sector", "topic_id", "topic_name"], as_index=False)
                .size()
                .rename(columns={"size": "patent_count"})
            )
            sector_n = merged.groupby("sector")["특허번호"].nunique().to_dict()
            topic_port["share"] = topic_port.apply(
                lambda r: round(r["patent_count"] / sector_n.get(r["sector"], 1), 4),
                axis=1,
            )
            topic_port = topic_port.sort_values(
                ["sector", "patent_count"], ascending=[True, False]
            )
            topic_port_path = OUT / "assignee_sector_topic_portfolio.csv"
            topic_port.to_csv(topic_port_path, index=False, encoding="utf-8-sig")
            print(f"Saved {topic_port_path}")

    # --- Top 15 per sector for narrative ---
    top_by_sector = {}
    for sector in ["big_tech", "big_pharma", "academia", "other", "unknown"]:
        part = top[top["sector"] == sector].head(15)
        top_by_sector[sector] = part.to_dict(orient="records")

    # Contrast table: big_tech vs big_pharma CPC shares
    def pivot_cpc(sector: str) -> dict:
        sub = cpc_port[cpc_port["sector"] == sector]
        return {r["subclass"]: r["share"] for _, r in sub.iterrows()}

    contrast = {
        "big_tech_cpc_share": pivot_cpc("big_tech"),
        "big_pharma_cpc_share": pivot_cpc("big_pharma"),
    }

    summary = {
        "source": "patents_preprocessed (ODP Q4 2025)",
        "date_range": f"{DATE_START} .. {DATE_END}",
        "n_patents": int(len(df)),
        "n_with_assignee": int((df["assignee_norm"] != "").sum()),
        "sector_totals": sector_totals.to_dict(orient="records"),
        "top_assignees_overall": top.head(25).to_dict(orient="records"),
        "top_by_sector": top_by_sector,
        "cpc_contrast_big_tech_vs_pharma": contrast,
        "outputs": {
            "assignee_top": str(top_path),
            "sector_yearly": str(sector_yearly_path),
            "cpc_portfolio": str(cpc_port_path),
            "topic_portfolio": str(OUT / "assignee_sector_topic_portfolio.csv")
            if len(topic_port)
            else None,
        },
    }
    summary_path = OUT / "assignee_comparison_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\n=== Sector totals ===")
    print(sector_totals.to_string(index=False))
    print("\n=== Top Big Tech ===")
    print(top[top["sector"] == "big_tech"].head(10).to_string(index=False))
    print("\n=== Top Big Pharma ===")
    print(top[top["sector"] == "big_pharma"].head(10).to_string(index=False))
    print("\n=== CPC share contrast (focal) ===")
    print(json.dumps(contrast, indent=2, ensure_ascii=False))
    print(f"\nSaved summary: {summary_path}")


if __name__ == "__main__":
    main()
