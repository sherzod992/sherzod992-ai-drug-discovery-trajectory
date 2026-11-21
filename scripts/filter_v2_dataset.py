"""
Phase 1: Build patents_preprocessed_v2.csv from existing preprocessed set.
- Deny display/OLED CPC & keywords
- Require AI signal AND (drug/pharma text OR biomedical CPC)
- Drop pure display assignees from the analysis set
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "data" / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

INPUT_CANDIDATES = [
    PROCESSED / "patents_preprocessed_strict.csv",
    PROCESSED / "patents_preprocessed.csv",
]

DENY_CPC_PREFIXES = (
    "H01L",   # semiconductors
    "G09G",   # display control
    "G09F",   # display advertising boards
    "C09K11", # luminescent materials
    "H05B33", # electroluminescent light sources
    "H10K",   # organic semiconductor devices (OLED family)
)

DENY_KEYWORDS = (
    "organic light emitting",
    "organic light-emitting",
    "oled",
    "electroluminescent",
    "electroluminescence",
    "display panel",
    "light emitting diode",
    "light-emitting diode",
    "pixel circuit",
    "organic electroluminescent",
)

AI_TERMS = (
    "artificial intelligence",
    "machine learning",
    "deep learning",
    "neural network",
    "neural networks",
    "generative ai",
    "large language model",
    "reinforcement learning",
    "graph neural",
    "transformer",
    "random forest",
    "support vector",
    "bayesian",
    "clustering",
    "embedding",
)

DRUG_TERMS = (
    "drug discovery",
    "drug design",
    "pharmaceutical",
    "molecule generation",
    "molecular design",
    "target identification",
    "clinical trial",
    "compound",
    "ligand",
    "protein",
    "antibody",
    "therapeutic",
    "pharmacokinet",
    "admet",
    "qsar",
    "docking",
    "de novo",
    "biomarker",
    "oncolog",
    "vaccine",
    "peptide",
)

BIO_CPC = ("A61K", "A61B", "A61M", "G16H", "G16C", "C07K", "C07D", "C12N", "C12Q")

DISPLAY_ASSIGNEES = (
    "SAMSUNG DISPLAY",
    "LG DISPLAY",
    "BOE TECHNOLOGY",
    "AU OPTRONICS",
    "JAPAN DISPLAY",
    "SHARP KABUSHIKI",
    "UNIVERSAL DISPLAY",
    "E INK",
)


def parse_codes(cell: str) -> list[str]:
    if not isinstance(cell, str) or not cell.strip():
        return []
    return [p.strip().upper() for p in re.split(r"[;|]", cell) if p.strip()]


def has_prefix(codes: list[str], prefixes: tuple[str, ...]) -> bool:
    return any(c.startswith(p) for c in codes for p in prefixes)


def deny_text(title: str, abstract: str) -> bool:
    t = f"{title or ''} {abstract or ''}".lower()
    return any(k in t for k in DENY_KEYWORDS)


def is_display_assignee(name: str) -> bool:
    n = (name or "").upper()
    return any(d in n for d in DISPLAY_ASSIGNEES)


def keep_row(row: pd.Series) -> bool:
    title = str(row.get("제목") or row.get("발명의명칭") or "")
    abstract = str(row.get("초록") or "")
    codes = parse_codes(str(row.get("CPC분류코드") or ""))
    assignee = str(row.get("출원인") or "")

    if is_display_assignee(assignee):
        return False
    if deny_text(title, abstract):
        return False
    # Drop if deny CPC present and no strong bio CPC
    if has_prefix(codes, DENY_CPC_PREFIXES) and not has_prefix(codes, BIO_CPC):
        return False
    # If deny CPC coexists with bio, still drop if OLED keywords already caught;
    # additional: if majority of codes are display-family, drop
    if has_prefix(codes, DENY_CPC_PREFIXES):
        deny_n = sum(1 for c in codes if any(c.startswith(p) for p in DENY_CPC_PREFIXES))
        if deny_n >= max(1, len(set(c[:4] for c in codes)) // 2):
            return False

    text = f"{title} {abstract}".lower()
    has_ai = any(term in text for term in AI_TERMS)
    has_drug = any(term in text for term in DRUG_TERMS)
    has_bio_cpc = has_prefix(codes, BIO_CPC)
    has_g06n = has_prefix(codes, ("G06N",))

    # Strict: AI signal AND (drug text OR biomedical CPC)
    # Allow G06N+bio CPC without explicit AI words (ML often coded only in CPC)
    if has_ai and (has_drug or has_bio_cpc):
        return True
    if has_g06n and has_bio_cpc:
        return True
    if has_drug and has_bio_cpc and has_ai:
        return True
    return False


def main() -> None:
    src = next((p for p in INPUT_CANDIDATES if p.exists()), None)
    if src is None:
        raise FileNotFoundError("No preprocessed CSV found. Run collect_stage1_data.py first.")

    df = pd.read_csv(src, encoding="utf-8-sig")
    n0 = len(df)
    print(f"Loaded {src.name}: {n0:,}")

    mask = df.apply(keep_row, axis=1)
    out = df[mask].copy()
    n1 = len(out)

    # Extra safety: remove residual OLED-ish rows
    title_col = "제목" if "제목" in out.columns else "발명의명칭"
    blob = (
        out[title_col].fillna("").astype(str) + " " + out["초록"].fillna("").astype(str)
    ).str.lower()
    oled_mask = blob.str.contains("oled|organic light emitting|electrolumines", regex=True)
    removed_oled = int(oled_mask.sum())
    out = out[~oled_mask]

    out_path = PROCESSED / "patents_preprocessed_v2.csv"
    out.to_csv(out_path, index=False, encoding="utf-8-sig")

    # Yearly counts
    dates = pd.to_datetime(out["출원일"], errors="coerce")
    years = dates.dt.year.value_counts().sort_index()
    summary = {
        "source": src.name,
        "n_before": n0,
        "n_after_filter": n1,
        "n_after_oled_sweep": int(len(out)),
        "removed_total": int(n0 - len(out)),
        "removed_oled_residual": removed_oled,
        "yearly_counts": {int(k): int(v) for k, v in years.items() if pd.notna(k)},
        "output": str(out_path),
    }
    summary_path = PROCESSED / "collection_summary_v2.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (RESULTS / "collection_summary_v2.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
