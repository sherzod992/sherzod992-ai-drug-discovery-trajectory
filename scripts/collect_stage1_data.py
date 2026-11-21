"""
Stage 1: PatentsView bulk (USPTO ODP Q4 2025 / through 2025-12-31)
+ Lens-style CPC·keyword·filing-date filter.
Outputs: data/processed/patents_raw.csv, patents_preprocessed.csv, collection_summary.txt

Requires env USPTO_API_KEY (ODP Manage API Key).
"""
from __future__ import annotations

import io
import os
import re
import zipfile
from collections import defaultdict
from datetime import date
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"
ODP_API = "https://api.uspto.gov/api/v1/datasets/products"
ODP_PRODUCT = "PVGPATDIS"
SOURCE_LABEL = "USPTO ODP PatentsView Q4 2025 (through 2025-12-31)"

DATE_START = date(2016, 1, 1)
DATE_END = date(2025, 12, 31)
# Core healthcare + AI CPC prefixes
CPC_PREFIXES = (
    "G16H",   # Healthcare informatics
    "G16C20", # Clinical trial / medical data
    "A61K",   # Pharmaceutical preparations
    # Extended: AI/ML architectures
    "G06N",   # AI / machine learning / neural networks
    "G06V",   # Image / video recognition (medical imaging)
    # Extended: Medical diagnosis & devices
    "A61B",   # Medical diagnosis / surgery / detectors
    "A61M",   # Medical devices / drug delivery
    # Extended: Chemistry / pharma compounds
    "C07K",   # Proteins / antibodies / peptides
    "C07D",   # Heterocyclic compounds / small molecules
    # Extended: Health informatics & bioinformatics
    "G16C",   # Health informatics
    "G16Z",   # Other healthcare IT
)

NEEDED_FILES = (
    "g_cpc_current.tsv.zip",
    "g_patent.tsv.zip",
    "g_application.tsv.zip",
    "g_assignee_disambiguated.tsv.zip",
    "g_patent_abstract.tsv.zip",
)


def odp_headers() -> dict[str, str] | None:
    """Return ODP headers when USPTO_API_KEY is set; otherwise None (local files only)."""
    key = (os.environ.get("USPTO_API_KEY") or "").strip()
    if not key:
        return None
    return {"x-api-key": key, "User-Agent": "patent-collect/1.0"}


def odp_file_url(fname: str) -> str:
    return f"{ODP_API}/files/{ODP_PRODUCT}/{fname}"


def ensure_local_files() -> None:
    missing = [f for f in NEEDED_FILES if not (RAW / f).exists()]
    if missing:
        raise SystemExit(
            "로컬 raw 파일이 없습니다: "
            + ", ".join(missing)
            + "\nhttps://data.uspto.gov/bulkdata/datasets/pvgpatdis 에서 받아 "
            + f"{RAW} 에 두거나 USPTO_API_KEY 를 설정하세요."
        )
    print(f"Using local files in {RAW}")

AI_TERMS = (
    "artificial intelligence",
    "machine learning",
    "deep learning",
    "neural network",
    "generative ai",
    "large language model",
    "llm",
    "convolutional neural",
    "reinforcement learning",
    "natural language processing",
    "computer vision",
    "predictive model",
    "classifier",
    "training data",
    "transformer",
    "bert",
    "gpt",
    "diffusion model",
    "gan",
    "generative adversarial",
    "graph neural",
    "graph convolutional",
    "computer-aided drug discovery",
    "cadd",
    "in silico",
    "virtual screening",
    "molecular docking",
    "molecular dynamics",
    "drug-target interaction",
    "dti",
    "de novo drug design",
    "retrosynthesis",
)

DRUG_TERMS = (
    "drug discovery",
    "drug design",
    "pharmaceutical",
    "molecule generation",
    "target identification",
    "clinical trial",
    "therapeutic",
    "compound",
    "small molecule",
    "biologics",
    "pharmacology",
    "antibody",
    "biomarker",
    "active pharmaceutical",
    "drug repurposing",
    "drug screening",
    "pharmacokinetics",
    "adme",
    "lead optimization",
    "preclinical",
    "drug candidate",
    "drug development",
    "medicinal chemistry",
    "pharmacodynamics",
    "toxicity",
    "drug metabolism",
    "active ingredient",
)

# 0 = no cap (keep full filtered set; avoids year/month gaps from stratified truncation)
MAX_ANALYSIS_ROWS = 0

LEGAL_BOILERPLATE = re.compile(
    r"disclosed herein|embodiments of the (present )?invention|"
    r"what is claimed is:|without departing from the spirit",
    re.I,
)


def expected_size(url: str, headers: dict[str, str] | None = None) -> int:
    r = requests.head(url, headers=headers, allow_redirects=True, timeout=60)
    r.raise_for_status()
    return int(r.headers.get("content-length") or 0)


def download(url: str, dest: Path, headers: dict[str, str] | None = None) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    need = expected_size(url, headers=headers)
    if dest.exists():
        # Force re-download when switching from Zenodo 2024 snapshot to ODP 2025
        # if local size no longer matches remote, or zip is corrupt.
        if need and dest.stat().st_size == need and zipfile.is_zipfile(dest):
            bad = zipfile.ZipFile(dest).testzip()
            if bad is None:
                print(f"  skip (complete): {dest.name}")
                return
        print(f"  remove incomplete/outdated: {dest.name}")
        dest.unlink()

    print(f"  download: {dest.name} ({need / 1e6:.1f} MB)")
    with requests.get(url, headers=headers, stream=True, timeout=120) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length") or need or 0)
        with open(dest, "wb") as f, tqdm(total=total, unit="B", unit_scale=True) as bar:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
                    bar.update(len(chunk))


def iter_tsv_zip(path: Path, usecols: list[str], chunksize: int = 250_000):
    with zipfile.ZipFile(path) as zf:
        name = [n for n in zf.namelist() if n.endswith(".tsv")][0]
        with zf.open(name) as raw:
            yield from pd.read_csv(
                io.TextIOWrapper(raw, encoding="utf-8", errors="replace"),
                sep="\t",
                usecols=usecols,
                dtype=str,
                chunksize=chunksize,
            )


def read_tsv_zip(path: Path, usecols: list[str] | None = None) -> pd.DataFrame:
    with zipfile.ZipFile(path) as zf:
        name = [n for n in zf.namelist() if n.endswith(".tsv")][0]
        with zf.open(name) as raw:
            return pd.read_csv(
                io.TextIOWrapper(raw, encoding="utf-8", errors="replace"),
                sep="\t",
                usecols=usecols,
                dtype=str,
                low_memory=False,
            )


def _cpc_label(subclass: str, group: str) -> str:
    grp = (group or "").strip()
    if grp:
        return grp
    return (subclass or "").strip()


def build_cpc_index(cpc_zip: Path) -> tuple[set[str], dict[str, list[str]]]:
    codes: dict[str, list[str]] = defaultdict(list)
    ids: set[str] = set()
    for chunk in tqdm(
        iter_tsv_zip(cpc_zip, ["patent_id", "cpc_subclass", "cpc_group"]),
        desc="cpc",
    ):
        sub = chunk["cpc_subclass"].fillna("")
        grp = chunk["cpc_group"].fillna("")
        mask = pd.Series(False, index=chunk.index)
        for p in CPC_PREFIXES:
            mask = mask | grp.str.startswith(p) | sub.str.startswith(p)
        hits = chunk.loc[mask]
        for pid, s, g in zip(
            hits["patent_id"], hits["cpc_subclass"], hits["cpc_group"]
        ):
            ids.add(pid)
            label = _cpc_label(s, g)
            if label and label not in codes[pid]:
                codes[pid].append(label)
    return ids, codes


def filter_applications(app_zip: Path, patent_ids: set[str]) -> pd.DataFrame:
    rows = []
    for chunk in iter_tsv_zip(app_zip, ["patent_id", "filing_date"]):
        sub = chunk[chunk["patent_id"].isin(patent_ids)]
        if not sub.empty:
            rows.append(sub)
    app = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    app = app.rename(columns={"filing_date": "application_date"})
    app["application_date"] = pd.to_datetime(app["application_date"], errors="coerce")
    app = app.dropna(subset=["application_date"])
    app = app[
        (app["application_date"].dt.date >= DATE_START)
        & (app["application_date"].dt.date <= DATE_END)
    ]
    return app.sort_values("application_date").groupby("patent_id", as_index=False).first()


def load_patents(patent_zip: Path, patent_ids: set[str]) -> pd.DataFrame:
    rows = []
    cols = ["patent_id", "patent_title", "patent_date"]
    for chunk in iter_tsv_zip(patent_zip, cols):
        sub = chunk[chunk["patent_id"].isin(patent_ids)]
        if not sub.empty:
            rows.append(sub)
    df = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=cols)
    df["patent_date"] = pd.to_datetime(df["patent_date"], errors="coerce")
    df["patent_number"] = df["patent_id"]
    df["num_us_patents_cited_by"] = 0
    return df


def load_assignees(assignee_zip: Path, patent_ids: set[str]) -> pd.DataFrame:
    rows = []
    for chunk in iter_tsv_zip(
        assignee_zip, ["patent_id", "disambig_assignee_organization"]
    ):
        sub = chunk[chunk["patent_id"].isin(patent_ids)]
        if not sub.empty:
            rows.append(sub)
    df = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    df["disambig_assignee_organization"] = (
        df["disambig_assignee_organization"].fillna("").str.strip()
    )
    df = df[df["disambig_assignee_organization"] != ""]
    return df.drop_duplicates(subset=["patent_id"], keep="first")


def load_abstracts_stream(abs_zip: Path, patent_ids: set[str]) -> pd.DataFrame:
    rows = []
    for chunk in tqdm(
        iter_tsv_zip(abs_zip, ["patent_id", "patent_abstract"]),
        desc="abstracts",
    ):
        sub = chunk[chunk["patent_id"].isin(patent_ids)]
        if not sub.empty:
            rows.append(sub)
    if not rows:
        return pd.DataFrame(columns=["patent_id", "patent_abstract"])
    return pd.concat(rows, ignore_index=True)


def _cpc_flags(cpc_list: list[str]) -> dict[str, bool]:
    """Check which expanded CPC prefixes are present in the patent's CPC list."""
    flags: dict[str, bool] = {p: False for p in CPC_PREFIXES}
    for c in cpc_list:
        for p in CPC_PREFIXES:
            if c.startswith(p):
                flags[p] = True
    return flags


def lens_style_match(title: str, abstract: str, cpc_list: list[str]) -> bool:
    """Expanded strict filter: (AI keyword OR drug keyword) AND relevant CPC code."""
    t = f"{title or ''} {abstract or ''}".lower()
    ab = (abstract or "").lower()
    has_ai = any(term in t for term in AI_TERMS)
    has_drug = any(term in ab for term in DRUG_TERMS)
    has_cpc = any(c.startswith(p) for c in cpc_list for p in CPC_PREFIXES)
    # Expanded: accept either AI or drug domain, as long as CPC is relevant
    return (has_ai or has_drug) and has_cpc


def broad_domain_match(title: str, abstract: str, cpc_list: list[str]) -> bool:
    """Broader Lens-aligned filter for 5k–15k analysis set (expanded CPC)."""
    t = f"{title or ''} {abstract or ''}".lower()
    ab = (abstract or "").lower()
    has_ai = any(term in t for term in AI_TERMS)
    has_drug = any(term in ab for term in DRUG_TERMS)
    flags = _cpc_flags(cpc_list)
    has_g16 = flags.get("G16H", False)
    has_g16c = flags.get("G16C20", False)
    has_a61k = flags.get("A61K", False)
    has_g06n = flags.get("G06N", False)
    has_g06v = flags.get("G06V", False)
    has_a61b = flags.get("A61B", False)
    has_c07k = flags.get("C07K", False)
    has_c07d = flags.get("C07D", False)

    # AI keyword + any relevant CPC
    if has_ai and (has_g16 or has_g16c or has_a61k or has_g06n or has_g06v or has_a61b or has_c07k or has_c07d):
        return True
    # Drug keyword + healthcare CPC (G16, A61)
    if has_drug and (has_g16 or has_g16c or has_a61k or has_a61b):
        return True
    # AI keyword + pharma CPC (A61K, C07K, C07D)
    if has_ai and (has_a61k or has_c07k or has_c07d):
        return True
    return False


def cpc_window_match(_title: str, abstract: str, cpc_list: list[str]) -> bool:
    if not (abstract or "").strip():
        return False
    return any(c.startswith(p) for c in cpc_list for p in CPC_PREFIXES)


def cap_stratified_by_year(df: pd.DataFrame, max_rows: int) -> pd.DataFrame:
    """Keep filing-year/month coverage when capping large CPC sets.

    Samples evenly across year-month bins (not year-end only) so early months
    of each year are not dropped.
    """
    if max_rows <= 0 or len(df) <= max_rows:
        return df.sort_values("application_date")
    tmp = df.copy()
    dates = pd.to_datetime(tmp["application_date"])
    tmp["_year"] = dates.dt.year
    tmp["_month"] = dates.dt.month
    bins = (
        tmp.dropna(subset=["_year", "_month"])
        .groupby(["_year", "_month"], sort=True)
        .size()
        .index.tolist()
    )
    if not bins:
        return df.sort_values("application_date").head(max_rows)
    per_bin = max_rows // len(bins)
    remainder = max_rows - per_bin * len(bins)
    parts: list[pd.DataFrame] = []
    for i, (y, m) in enumerate(bins):
        chunk = tmp[(tmp["_year"] == y) & (tmp["_month"] == m)].sample(
            frac=1.0, random_state=42
        )
        n = per_bin + (1 if i < remainder else 0)
        if n > 0:
            parts.append(chunk.head(n))
    out = pd.concat(parts, ignore_index=True) if parts else tmp.head(0)
    if len(out) < max_rows:
        used = set(out["patent_id"])
        extra = (
            tmp[~tmp["patent_id"].isin(used)]
            .sample(frac=1.0, random_state=42)
            .head(max_rows - len(out))
        )
        out = pd.concat([out, extra], ignore_index=True)
    return out.sort_values("application_date").drop(
        columns=["_year", "_month"], errors="ignore"
    )


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    out = df.drop_duplicates(subset=["특허번호"], keep="first")
    out = out.dropna(subset=["초록", "CPC분류코드"])
    out = out[out["초록"].str.len() > 80]
    out = out[out["CPC분류코드"].str.len() > 3]
    out["초록"] = out["초록"].apply(
        lambda t: LEGAL_BOILERPLATE.sub(" ", t) if isinstance(t, str) else t
    )
    out["출원인"] = (
        out["출원인"].fillna("").str.replace(r"\s+", " ", regex=True).str.strip().str.upper()
    )
    return out.reset_index(drop=True)


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)

    headers = odp_headers()
    if headers:
        meta = requests.get(
            f"{ODP_API}/{ODP_PRODUCT}",
            headers=headers,
            params={"includeFiles": "true", "latest": "true"},
            timeout=60,
        )
        meta.raise_for_status()
        print(f"ODP product: {ODP_PRODUCT} ({SOURCE_LABEL})")
        for fname in NEEDED_FILES:
            download(odp_file_url(fname), RAW / fname, headers=headers)
    else:
        print(f"No USPTO_API_KEY — {SOURCE_LABEL} via local files")
        ensure_local_files()


    print("CPC filter …")
    cpc_ids, code_map = build_cpc_index(RAW / "g_cpc_current.tsv.zip")
    print(f"  CPC-matched: {len(cpc_ids):,}")

    print("Application date filter …")
    apps = filter_applications(RAW / "g_application.tsv.zip", cpc_ids)
    target_ids = set(apps["patent_id"])
    print(f"  Filing window: {len(target_ids):,}")

    patents = load_patents(RAW / "g_patent.tsv.zip", target_ids)
    assignees = load_assignees(
        RAW / "g_assignee_disambiguated.tsv.zip", target_ids
    )
    abstracts = load_abstracts_stream(
        RAW / "g_patent_abstract.tsv.zip", target_ids
    )

    df = apps.merge(patents, on="patent_id", how="inner")
    df = df.merge(abstracts, on="patent_id", how="inner")
    df = df.merge(assignees, on="patent_id", how="left")
    df["cpc_codes"] = df["patent_id"].map(lambda pid: ";".join(code_map.get(pid, [])))
    df["cpc_list"] = df["patent_id"].map(lambda pid: code_map.get(pid, []))

    before = len(df)
    strict = df[
        df.apply(
            lambda r: lens_style_match(
                r.get("patent_title"), r.get("patent_abstract"), r["cpc_list"]
            ),
            axis=1,
        )
    ]
    broad = df[
        df.apply(
            lambda r: broad_domain_match(
                r.get("patent_title"), r.get("patent_abstract"), r["cpc_list"]
            ),
            axis=1,
        )
    ]
    cpc_only = df[
        df.apply(
            lambda r: cpc_window_match(
                r.get("patent_title"), r.get("patent_abstract"), r["cpc_list"]
            ),
            axis=1,
        )
    ]
    print(f"  strict: {len(strict):,}  broad: {len(broad):,}  cpc+abstract: {len(cpc_only):,}")

    # Primary set: prefer broad; if <5k use CPC+abstract stratified cap to MAX_ANALYSIS_ROWS
    if len(broad) >= 5_000:
        df = broad if len(broad) <= MAX_ANALYSIS_ROWS else cap_stratified_by_year(
            broad, MAX_ANALYSIS_ROWS
        )
        label = "broad" if len(broad) <= MAX_ANALYSIS_ROWS else "broad (stratified cap)"
    else:
        df = cap_stratified_by_year(cpc_only, min(MAX_ANALYSIS_ROWS, len(cpc_only)))
        label = "cpc+abstract (stratified cap)"
    print(f"  using {label}: {len(df):,}")

    strict_path = OUT / "patents_preprocessed_strict.csv"
    strict_out = strict.rename(
        columns={
            "patent_number": "특허번호",
            "patent_title": "제목",
            "patent_abstract": "초록",
            "application_date": "출원일",
            "disambig_assignee_organization": "출원인",
            "cpc_codes": "CPC분류코드",
            "num_us_patents_cited_by": "피인용수",
            "patent_date": "등록일",
        }
    )
    preprocess(strict_out).to_csv(strict_path, index=False, encoding="utf-8-sig")
    print(f"  strict subset saved: {strict_path} ({len(strict):,} rows)")

    df = df.rename(
        columns={
            "patent_number": "특허번호",
            "patent_title": "제목",
            "patent_abstract": "초록",
            "application_date": "출원일",
            "disambig_assignee_organization": "출원인",
            "cpc_codes": "CPC분류코드",
            "num_us_patents_cited_by": "피인용수",
            "patent_date": "등록일",
        }
    )
    cols = [
        "특허번호",
        "제목",
        "초록",
        "출원일",
        "등록일",
        "출원인",
        "CPC분류코드",
        "피인용수",
    ]
    df = df[cols].sort_values("출원일")

    raw_path = OUT / "patents_raw.csv"
    df.to_csv(raw_path, index=False, encoding="utf-8-sig")
    print(f"raw saved: {raw_path} ({len(df):,} rows)")

    clean = preprocess(df)
    clean_path = OUT / "patents_preprocessed.csv"
    clean.to_csv(clean_path, index=False, encoding="utf-8-sig")
    print(f"preprocessed: {clean_path} ({len(clean):,} rows)")

    yearly = (
        clean.assign(연도=clean["출원일"].dt.year)
        .groupby("연도")
        .size()
        .to_dict()
    )
    summary_lines = [
        f"source: {SOURCE_LABEL}",
        f"date_range: {DATE_START} .. {DATE_END}",
        f"cpc_prefixes: {list(CPC_PREFIXES)}",
        f"cpc_matched: {len(cpc_ids)}",
        f"filing_window: {len(target_ids)}",
        f"merged_with_abstract: {before}",
        f"filter_strict: {len(strict)}",
        f"filter_broad: {len(broad)}",
        f"filter_cpc_abstract: {len(cpc_only)}",
        f"analysis_set_rows: {len(df)}",
        f"preprocessed_rows: {len(clean)}",
        f"yearly_counts: {yearly}",
        f"raw_csv: {raw_path}",
        f"preprocessed_csv: {clean_path}",
        f"strict_csv: {strict_path}",
    ]
    (OUT / "collection_summary.txt").write_text(
        "\n".join(summary_lines), encoding="utf-8"
    )


if __name__ == "__main__":
    main()