"""
prepare_data.py — download the IBM AMLSim sample and build the network files the app uses.

Run once:   python prepare_data.py
(The app also calls build_all() automatically if the data folder is empty.)

Source: IBM AMLSim (Apache-2.0), sample "20K_fanin200cycle200"
https://github.com/IBM/AMLSim  ->  sample/20K_fanin200cycle200.tgz
20,000 bank accounts, 120,558 transfers, 200 cycle + 200 fan-in laundering patterns.
"""
import io
import tarfile
import urllib.request
from pathlib import Path

import networkx as nx
import pandas as pd

URL = "https://raw.githubusercontent.com/IBM/AMLSim/master/sample/20K_fanin200cycle200.tgz"
DATA = Path(__file__).parent / "data"
RAW_NODES, RAW_TX = DATA / "raw_nodes.csv", DATA / "raw_transactions.csv"
NODES, EDGES = DATA / "nodes.csv", DATA / "edges.csv"


def download():
    """Download the tgz from GitHub and keep the two CSVs we need."""
    DATA.mkdir(exist_ok=True)
    if RAW_NODES.exists() and RAW_TX.exists():
        return
    print("Downloading", URL)
    blob = urllib.request.urlopen(URL, timeout=120).read()
    with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as tar:
        for m in tar.getmembers():
            if m.name.endswith("nodes.csv"):
                RAW_NODES.write_bytes(tar.extractfile(m).read())
            elif m.name.endswith("transactions.csv"):
                RAW_TX.write_bytes(tar.extractfile(m).read())


def build_all():
    download()
    nodes = pd.read_csv(RAW_NODES)
    tx = pd.read_csv(RAW_TX)

    # 1) Edge list: one directed edge per (sender -> receiver) pair, weighted by money and count
    edges = (tx.groupby(["sourceNodeId", "targetNodeId"])
               .agg(amount=("value", "sum"), n_tx=("value", "size"))
               .reset_index()
               .rename(columns={"sourceNodeId": "source", "targetNodeId": "target"}))
    edges = edges[edges.source != edges.target]          # drop 15 self-transfers
    edges["amount"] = edges["amount"].round(2)

    # 2) Graph
    G = nx.from_pandas_edgelist(edges, "source", "target", ["amount", "n_tx"], create_using=nx.DiGraph)
    G.add_nodes_from(nodes.nodeid)                        # keep isolated accounts too
    U = G.to_undirected()

    # 3) Structural measures (computed ONCE on the full graph, then filtered in the app)
    print("Computing measures ...")
    m = pd.DataFrame({"account": list(G.nodes())})
    m["in_degree"] = m.account.map(dict(G.in_degree()))
    m["out_degree"] = m.account.map(dict(G.out_degree()))
    m["degree"] = m.in_degree + m.out_degree
    m["money_in"] = m.account.map(dict(G.in_degree(weight="amount"))).round(2)
    m["money_out"] = m.account.map(dict(G.out_degree(weight="amount"))).round(2)
    m["pagerank"] = m.account.map(nx.pagerank(G, alpha=0.85))
    # Betweenness is O(N*M): use Brandes with 500 sampled sources (an approximation, stated in the app)
    m["betweenness"] = m.account.map(nx.betweenness_centrality(G, k=500, seed=42))
    m["clustering"] = m.account.map(nx.clustering(U))
    comms = nx.community.louvain_communities(U, seed=42)
    comms = sorted(comms, key=len, reverse=True)
    cid = {n: i for i, c in enumerate(comms) for n in c}
    m["community"] = m.account.map(cid)
    m = m.merge(nodes.rename(columns={"nodeid": "account", "isFraud": "laundering"}), on="account")
    m["laundering"] = m.laundering.astype(int)
    m["modularity"] = nx.community.modularity(U, comms)  # same value on every row (graph-level)

    m.to_csv(NODES, index=False)
    edges.to_csv(EDGES, index=False)
    print(f"Saved {NODES} ({len(m)} accounts) and {EDGES} ({len(edges)} edges)")


if __name__ == "__main__":
    build_all()
