"""
Step 3: CPC co-occurrence network analysis (2016–2025).
Inputs:  data/processed/patents_preprocessed.csv
Outputs: data/processed/cpc_*.csv|json  (+ root summaries for convenience)
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import date
from itertools import combinations
from pathlib import Path

import networkx as nx
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
OUT = PROCESSED
OUT.mkdir(parents=True, exist_ok=True)

DATE_START = date(2016, 1, 1)
DATE_END = date(2025, 12, 31)

# Focal subclasses for AI × biomedical convergence narrative
FOCAL_SUBCLASSES = (
    "G16H",  # healthcare informatics
    "G16C",  # computational chemistry / bioinformatics
    "G16Z",
    "G06N",  # AI / ML
    "G06V",  # image recognition
    "A61K",  # pharmaceutical preparations
    "A61B",  # diagnosis / surgery
    "A61M",  # medical devices
    "C07K",  # peptides / proteins
    "C07D",  # heterocyclic compounds
)

INPUT_CANDIDATES = [
    PROCESSED / "patents_preprocessed.csv",
    PROCESSED / "patents_preprocessed_strict.csv",
]


def subclass_of(code: str) -> str:
    """Map CPC symbol to 4-char subclass (e.g. G06N20/00 -> G06N)."""
    c = (code or "").strip().upper().replace(" ", "")
    if not c:
        return ""
    # already short
    if len(c) <= 4:
        return c
    # G06N20/00 or G06N3/08
    return c[:4]


def parse_codes(cell: str) -> list[str]:
    if not isinstance(cell, str) or not cell.strip():
        return []
    parts = [p.strip() for p in cell.replace("|", ";").split(";") if p.strip()]
    # unique preserve order
    seen: set[str] = set()
    out: list[str] = []
    for p in parts:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def load_df() -> pd.DataFrame:
    for path in INPUT_CANDIDATES:
        if path.exists():
            df = pd.read_csv(path, encoding="utf-8-sig")
            print(f"Loaded: {path.name} ({len(df):,} rows)")
            return df
    raise FileNotFoundError("No preprocessed patent CSV found. Run stage1 first.")


def build_pair_counts(code_lists: list[list[str]]) -> Counter[tuple[str, str]]:
    pairs: Counter[tuple[str, str]] = Counter()
    for codes in code_lists:
        uniq = sorted(set(codes))
        if len(uniq) < 2:
            continue
        for a, b in combinations(uniq, 2):
            pairs[(a, b)] += 1
    return pairs


def graph_from_pairs(pairs: Counter[tuple[str, str]], min_weight: int = 2) -> nx.Graph:
    g = nx.Graph()
    for (a, b), w in pairs.items():
        if w >= min_weight:
            g.add_edge(a, b, weight=w)
    return g


def centrality_table(g: nx.Graph) -> pd.DataFrame:
    if g.number_of_nodes() == 0:
        return pd.DataFrame(
            columns=["node", "degree", "weighted_degree", "betweenness", "eigenvector"]
        )
    deg = dict(g.degree())
    wdeg = dict(g.degree(weight="weight"))
    # approximate betweenness for speed on larger graphs
    k = min(200, g.number_of_nodes())
    btw = nx.betweenness_centrality(g, weight="weight", k=k, seed=42)
    try:
        eig = nx.eigenvector_centrality_numpy(g, weight="weight")
    except Exception:
        eig = {n: 0.0 for n in g.nodes()}
    rows = [
        {
            "node": n,
            "degree": deg.get(n, 0),
            "weighted_degree": wdeg.get(n, 0),
            "betweenness": btw.get(n, 0.0),
            "eigenvector": float(eig.get(n, 0.0)),
        }
        for n in g.nodes()
    ]
    return pd.DataFrame(rows).sort_values("weighted_degree", ascending=False)


def yearly_focal_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Per-year focal subclass co-occurrence edges."""
    rows = []
    for year, sub in df.groupby("연도"):
        node_count: Counter[str] = Counter()
        edge_count: Counter[tuple[str, str]] = Counter()
        for codes in sub["subclasses"]:
            focal = sorted({c for c in codes if c in FOCAL_SUBCLASSES})
            for c in focal:
                node_count[c] += 1
            for a, b in combinations(focal, 2):
                edge_count[(a, b)] += 1
        for (a, b), w in edge_count.items():
            rows.append(
                {
                    "연도": int(year),
                    "source": a,
                    "target": b,
                    "co_occurrence": w,
                    "source_patents": node_count[a],
                    "target_patents": node_count[b],
                }
            )
    return pd.DataFrame(rows)


def yearly_focal_centrality(df: pd.DataFrame) -> pd.DataFrame:
    """Yearly weighted degree among focal subclasses only."""
    rows = []
    for year, sub in df.groupby("연도"):
        pairs = build_pair_counts(sub["subclasses"].tolist())
        # keep only focal–focal edges
        focal_pairs: Counter[tuple[str, str]] = Counter()
        for (a, b), w in pairs.items():
            if a in FOCAL_SUBCLASSES and b in FOCAL_SUBCLASSES:
                focal_pairs[(a, b)] += w
        g = graph_from_pairs(focal_pairs, min_weight=1)
        # ensure all focal nodes that appear that year exist
        for codes in sub["subclasses"]:
            for c in codes:
                if c in FOCAL_SUBCLASSES:
                    g.add_node(c)
        wdeg = dict(g.degree(weight="weight"))
        for node in FOCAL_SUBCLASSES:
            if node in g:
                rows.append(
                    {
                        "연도": int(year),
                        "subclass": node,
                        "weighted_degree": wdeg.get(node, 0),
                        "degree": g.degree(node) if node in g else 0,
                        "n_patents": sum(
                            1 for codes in sub["subclasses"] if node in codes
                        ),
                    }
                )
    return pd.DataFrame(rows)


def main() -> None:
    df = load_df()
    df = df.copy()
    df["출원일"] = pd.to_datetime(df["출원일"], errors="coerce")
    df = df.dropna(subset=["출원일", "CPC분류코드"])
    df["연도"] = df["출원일"].dt.year
    df = df[(df["연도"] >= DATE_START.year) & (df["연도"] <= DATE_END.year)]
    print(f"After date filter ({DATE_START.year}-{DATE_END.year}): {len(df):,}")

    df["cpc_list"] = df["CPC분류코드"].apply(parse_codes)
    df = df[df["cpc_list"].map(len) > 0]
    df["subclasses"] = df["cpc_list"].apply(
        lambda xs: sorted({subclass_of(x) for x in xs if subclass_of(x)})
    )
    print(f"Patents with CPC: {len(df):,}")

    # --- Code frequency ---
    code_freq = Counter()
    code_years: dict[str, set[int]] = defaultdict(set)
    for year, codes in zip(df["연도"], df["cpc_list"]):
        for c in set(codes):
            code_freq[c] += 1
            code_years[c].add(int(year))
    common = (
        pd.DataFrame(
            [
                {
                    "CPC": c,
                    "total_count": n,
                    "appearance_years": len(code_years[c]),
                    "subclass": subclass_of(c),
                }
                for c, n in code_freq.most_common()
            ]
        )
    )
    common_path = OUT / "common_cpcs.csv"
    common.to_csv(common_path, index=False, encoding="utf-8-sig")
    # also mirror to project root (legacy location)
    common[["CPC", "total_count", "appearance_years"]].to_csv(
        ROOT / "common_cpcs.csv", index=False, encoding="utf-8-sig"
    )
    print(f"Saved {common_path} ({len(common):,} codes)")

    # --- Subclass co-occurrence network ---
    subclass_lists = df["subclasses"].tolist()
    subclass_pairs = build_pair_counts(subclass_lists)
    g_sub = graph_from_pairs(subclass_pairs, min_weight=5)
    print(
        f"Subclass network: {g_sub.number_of_nodes():,} nodes, "
        f"{g_sub.number_of_edges():,} edges (min_weight=5)"
    )

    edges_sub = pd.DataFrame(
        [
            {"source": a, "target": b, "weight": w}
            for (a, b), w in subclass_pairs.most_common()
            if w >= 5
        ]
    )
    edges_path = OUT / "cpc_subclass_edges.csv"
    edges_sub.to_csv(edges_path, index=False, encoding="utf-8-sig")

    cent = centrality_table(g_sub)
    cent_path = OUT / "cpc_subclass_centrality.csv"
    cent.to_csv(cent_path, index=False, encoding="utf-8-sig")
    print(f"Saved centrality: {cent_path}")

    # Focal subgraph centrality
    focal_nodes = [n for n in FOCAL_SUBCLASSES if n in g_sub]
    g_focal = g_sub.subgraph(focal_nodes).copy()
    focal_cent = centrality_table(g_focal)
    focal_cent_path = OUT / "cpc_focal_centrality.csv"
    focal_cent.to_csv(focal_cent_path, index=False, encoding="utf-8-sig")

    # --- Yearly focal dynamics ---
    y_edges = yearly_focal_stats(df)
    y_edges_path = OUT / "cpc_yearly_focal_edges.csv"
    y_edges.to_csv(y_edges_path, index=False, encoding="utf-8-sig")

    y_cent = yearly_focal_centrality(df)
    y_cent_path = OUT / "cpc_yearly_focal_centrality.csv"
    y_cent.to_csv(y_cent_path, index=False, encoding="utf-8-sig")
    print(f"Saved yearly focal centrality: {y_cent_path}")

    # Top focal–focal edges overall
    focal_edge_totals = (
        y_edges[y_edges["source"] != y_edges["target"]]
        .groupby(["source", "target"], as_index=False)["co_occurrence"]
        .sum()
        .sort_values("co_occurrence", ascending=False)
    )

    # Summary JSON
    top_nodes = cent.head(15)[["node", "weighted_degree", "betweenness", "eigenvector"]]
    top_focal = focal_cent.sort_values("weighted_degree", ascending=False)
    summary = {
        "source": "patents_preprocessed (ODP Q4 2025)",
        "date_range": f"{DATE_START} .. {DATE_END}",
        "n_patents": int(len(df)),
        "n_unique_cpc_codes": int(len(common)),
        "subclass_network": {
            "nodes": int(g_sub.number_of_nodes()),
            "edges": int(g_sub.number_of_edges()),
            "min_edge_weight": 5,
        },
        "focal_subclasses": list(FOCAL_SUBCLASSES),
        "top_subclass_by_weighted_degree": top_nodes.to_dict(orient="records"),
        "focal_centrality": top_focal.to_dict(orient="records"),
        "top_focal_edges": focal_edge_totals.head(20).to_dict(orient="records"),
        "outputs": {
            "common_cpcs": str(common_path),
            "subclass_edges": str(edges_path),
            "subclass_centrality": str(cent_path),
            "focal_centrality": str(focal_cent_path),
            "yearly_focal_edges": str(y_edges_path),
            "yearly_focal_centrality": str(y_cent_path),
        },
    }
    summary_path = OUT / "cpc_network_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (ROOT / "cpc_network_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # Compact yearly summary: G16H / G06N / A61K weighted degree over time
    focus_track = ["G16H", "G16C", "G06N", "A61K", "C07K", "C07D"]
    yearly = {
        sc: {
            int(r["연도"]): int(r["weighted_degree"])
            for _, r in y_cent[y_cent["subclass"] == sc].iterrows()
        }
        for sc in focus_track
    }
    yearly_path = OUT / "cpc_yearly_summary.json"
    yearly_path.write_text(
        json.dumps(yearly, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (ROOT / "cpc_yearly_summary.json").write_text(
        json.dumps(yearly, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\n=== Focal centrality (weighted degree) ===")
    print(top_focal.to_string(index=False))
    print("\n=== Top focal co-occurrence edges ===")
    print(focal_edge_totals.head(12).to_string(index=False))
    print(f"\nSaved summary: {summary_path}")


if __name__ == "__main__":
    main()
