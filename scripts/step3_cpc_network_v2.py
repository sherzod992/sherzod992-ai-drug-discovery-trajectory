"""Phase 3: Period CPC metrics on patents_preprocessed_v2.csv"""
from __future__ import annotations

import json
from collections import Counter
from itertools import combinations
from pathlib import Path

import networkx as nx
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "data" / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

INPUT = PROCESSED / "patents_preprocessed_v2.csv"
FOCAL = ("G16H", "G16C", "G16Z", "G06N", "G06V", "A61K", "A61B", "A61M", "C07K", "C07D")
PERIODS = {
    "P1_2016_2019": range(2016, 2020),
    "P2_2020_2022": range(2020, 2023),
    "P3_2023_2025": range(2023, 2026),
}


def subclass_of(code: str) -> str:
    c = (code or "").strip().upper().replace(" ", "")
    return c[:4] if c else ""


def parse_codes(cell: str) -> list[str]:
    if not isinstance(cell, str) or not cell.strip():
        return []
    return [p.strip() for p in cell.replace("|", ";").split(";") if p.strip()]


def build_graph(subclass_lists: list[list[str]], min_w: int = 3) -> nx.Graph:
    pairs: Counter = Counter()
    for codes in subclass_lists:
        uniq = sorted({c for c in codes if c})
        for a, b in combinations(uniq, 2):
            pairs[(a, b)] += 1
    g = nx.Graph()
    for (a, b), w in pairs.items():
        if w >= min_w:
            g.add_edge(a, b, weight=w)
    return g, pairs


def metrics(g: nx.Graph) -> dict:
    if g.number_of_nodes() == 0:
        return {
            "nodes": 0, "edges": 0, "density": 0.0,
            "avg_clustering": 0.0, "n_components": 0,
        }
    return {
        "nodes": g.number_of_nodes(),
        "edges": g.number_of_edges(),
        "density": round(nx.density(g), 6),
        "avg_clustering": round(nx.average_clustering(g, weight=None), 6),
        "n_components": nx.number_connected_components(g),
    }


def centrality(g: nx.Graph) -> pd.DataFrame:
    if g.number_of_nodes() == 0:
        return pd.DataFrame(columns=["node", "weighted_degree", "betweenness", "degree"])
    wdeg = dict(g.degree(weight="weight"))
    btw = nx.betweenness_centrality(g, weight="weight", normalized=True)
    rows = [
        {
            "node": n,
            "weighted_degree": wdeg.get(n, 0),
            "betweenness": btw.get(n, 0.0),
            "degree": g.degree(n),
        }
        for n in g.nodes()
    ]
    return pd.DataFrame(rows).sort_values("weighted_degree", ascending=False)


def main() -> None:
    df = pd.read_csv(INPUT, encoding="utf-8-sig")
    df["출원일"] = pd.to_datetime(df["출원일"], errors="coerce")
    df = df.dropna(subset=["출원일", "CPC분류코드"])
    df["연도"] = df["출원일"].dt.year
    df["cpc_list"] = df["CPC분류코드"].apply(parse_codes)
    df["subclasses"] = df["cpc_list"].apply(
        lambda xs: sorted({subclass_of(x) for x in xs if subclass_of(x)})
    )
    print(f"Patents: {len(df):,}")

    period_metric_rows = []
    period_cent_rows = []
    period_edge_rows = []

    for pname, years in PERIODS.items():
        sub = df[df["연도"].isin(years)]
        lists = sub["subclasses"].tolist()
        g_all, pairs = build_graph(lists, min_w=3)
        # focal subgraph
        focal_nodes = [n for n in FOCAL if n in g_all]
        g_focal = g_all.subgraph(focal_nodes).copy()
        m = metrics(g_focal)
        m.update({"period": pname, "n_patents": int(len(sub)), "scope": "focal"})
        period_metric_rows.append(m)

        cent = centrality(g_focal)
        for _, r in cent.iterrows():
            period_cent_rows.append(
                {
                    "period": pname,
                    "node": r["node"],
                    "weighted_degree": r["weighted_degree"],
                    "betweenness": r["betweenness"],
                    "degree": r["degree"],
                }
            )

        # key bridge edges among focal
        for (a, b), w in pairs.items():
            if a in FOCAL and b in FOCAL and a != b:
                period_edge_rows.append(
                    {"period": pname, "source": a, "target": b, "co_occurrence": w}
                )

        print(pname, "patents", len(sub), "focal", m)

    metrics_df = pd.DataFrame(period_metric_rows)
    cent_df = pd.DataFrame(period_cent_rows)
    edges_df = (
        pd.DataFrame(period_edge_rows)
        .groupby(["period", "source", "target"], as_index=False)["co_occurrence"]
        .sum()
        .sort_values(["period", "co_occurrence"], ascending=[True, False])
    )

    for name, frame in [
        ("cpc_period_metrics_v2.csv", metrics_df),
        ("cpc_period_centrality_v2.csv", cent_df),
        ("cpc_period_edges_v2.csv", edges_df),
    ]:
        frame.to_csv(PROCESSED / name, index=False, encoding="utf-8-sig")
        frame.to_csv(RESULTS / name, index=False, encoding="utf-8-sig")

    # overall focal summary too
    g_all, pairs = build_graph(df["subclasses"].tolist(), min_w=3)
    g_focal = g_all.subgraph([n for n in FOCAL if n in g_all]).copy()
    summary = {
        "version": "v2",
        "n_patents": int(len(df)),
        "periods": metrics_df.to_dict(orient="records"),
        "overall_focal_centrality": centrality(g_focal).to_dict(orient="records"),
        "top_overall_focal_edges": sorted(
            [
                {"source": a, "target": b, "co_occurrence": w}
                for (a, b), w in pairs.items()
                if a in FOCAL and b in FOCAL
            ],
            key=lambda x: -x["co_occurrence"],
        )[:20],
    }
    (PROCESSED / "cpc_network_summary_v2.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (RESULTS / "cpc_network_summary_v2.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(metrics_df.to_string(index=False))
    print("Saved period CPC v2 outputs")


if __name__ == "__main__":
    main()
