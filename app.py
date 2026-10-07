"""
AML Network Explorer — DATS 6401 Week 6 homework (Graph & Network Data)

Bank-transfer network from IBM AMLSim: 20,000 accounts, 120,558 transfers,
1,804 accounts involved in simulated money-laundering patterns (fan-in & cycles).

Run:  streamlit run app.py
"""
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from pyvis.network import Network

st.set_page_config(page_title="AML Network Explorer", page_icon="🏦", layout="wide")

DATA = Path(__file__).parent / "data"

# ---------- colour system (Okabe–Ito: colour-blind safe) ----------
C_NORMAL, C_LAUNDER = "#0072B2", "#D55E00"           # blue = normal, vermillion = laundering
COMM_PAL = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00", "#F0E442", "#000000"]
C_OTHER = "#BBBBBB"
MEASURES = {
    "degree": "Degree (number of counterparties, in + out)",
    "in_degree": "In-degree (number of senders)",
    "out_degree": "Out-degree (number of receivers)",
    "pagerank": "PageRank (money-flow importance)",
    "betweenness": "Betweenness (broker / pass-through)",
    "money_in": "Money received ($)",
}


def show_html(html, height):
    """st.iframe (Streamlit ≥ 1.5x) replaces the deprecated components.v1.html; fall back for older versions."""
    if hasattr(st, "iframe"):
        st.iframe(html, height=height)
    else:
        import streamlit.components.v1 as components
        components.html(html, height=height)


# ---------- data ----------
@st.cache_data(show_spinner="Loading the bank-transfer network …")
def load_data():
    if not (DATA / "nodes.csv").exists():
        import prepare_data                        # downloads AMLSim sample + computes measures (~2 min)
        prepare_data.build_all()
    nodes = pd.read_csv(DATA / "nodes.csv")
    edges = pd.read_csv(DATA / "edges.csv")
    return nodes, edges


@st.cache_resource(show_spinner="Building the graph …")
def build_graph(_edges, n_nodes):
    G = nx.from_pandas_edgelist(_edges, "source", "target", ["amount", "n_tx"], create_using=nx.DiGraph)
    return G


nodes, edges = load_data()
G = build_graph(edges, len(nodes))
info = nodes.set_index("account")
BASE = nodes.laundering.mean()
Q = float(nodes.modularity.iloc[0])
N_COMM = nodes.community.nunique()
TOP_COMMS = nodes.community.value_counts().index[:8].tolist()     # only 8 hues are distinguishable


def lift_table(ks=(25, 50, 100, 200, 500)):
    rows = []
    for m in MEASURES:
        for k in ks:
            rows.append({"measure": m, "k": k, "share": nodes.nlargest(k, m).laundering.mean()})
    return pd.DataFrame(rows)


LIFT = lift_table()
share_top50_deg = LIFT.query("measure=='degree' and k==50").share.iloc[0]
tri_l = (nodes[nodes.laundering == 1].clustering > 0).mean()
tri_n = (nodes[nodes.laundering == 0].clustering > 0).mean()
comm_share = nodes.groupby("community").laundering.mean()
big_comm_share = nodes.groupby("community").agg(n=("account", "size"), s=("laundering", "mean")).query("n>=100").s

# ---------- sidebar controls ----------
st.sidebar.title("🏦 Controls")
view = st.sidebar.radio(
    "What to show",
    ["Top-k accounts (centrality filter)", "Ego network of one account", "Communities (aggregated)"],
    key="view",
    help="20,000 accounts cannot be drawn readably (hairball). Each view is a different, stated way of choosing a readable part.",
)
size_by = st.sidebar.selectbox("Node SIZE encodes", list(MEASURES), format_func=lambda m: MEASURES[m], key="size_by")
if view != "Communities (aggregated)":
    colour_by = st.sidebar.radio("Node COLOUR encodes", ["AML label", "Louvain community"], key="colour_by")
    layout = st.sidebar.selectbox("Layout", ["Force-directed", "Circular", "Shell (laundering inside)"], key="layout")

if view == "Top-k accounts (centrality filter)":
    filter_by = st.sidebar.selectbox("Keep the top-k accounts by", list(MEASURES), format_func=lambda m: MEASURES[m],
                                     key="filter_by")
    k = st.sidebar.slider("k (number of accounts)", 25, 400, 150, step=25, key="k")
    hide_isolated = st.sidebar.checkbox("Hide accounts with no link inside the view", value=True, key="hide_iso")
elif view == "Ego network of one account":
    hubs = nodes.nlargest(40, "degree").account.tolist()
    acct = st.sidebar.selectbox("Account (40 busiest listed; ◆ = laundering)", hubs,
                                format_func=lambda a: f"{a}  {'◆' if info.at[a, 'laundering'] else '●'}  deg {info.at[a, 'degree']}",
                                key="acct")
    typed = st.sidebar.text_input("…or type any account id (0–19999)", "", key="typed",
                                  placeholder="leave empty to use the list")
    if typed.strip().isdigit() and int(typed) in info.index:
        acct = int(typed)
    radius = st.sidebar.radio("Hops from the account", [1, 2], horizontal=True, key="radius")
    max_nodes = st.sidebar.slider("Max accounts drawn", 50, 500, 300, step=50, key="max_nodes",
                                  help="2-hop neighbourhoods explode; keep the most central ones.")
else:
    min_flow = st.sidebar.slider("Hide community links carrying less than ($ thousands)", 0, 300, 50, step=10, key="min_flow")

with st.sidebar.expander("ℹ️ How to read the graph"):
    st.markdown(
        "- **Drag** a node; neighbours follow (force layout).\n"
        "- **Hover** a node for its numbers.\n"
        "- **Scroll** to zoom.\n"
        "- **Arrows** = direction of money; **edge width** = $ amount (log)."
    )


# ---------- build the view subgraph ----------
def view_subgraph():
    if view == "Top-k accounts (centrality filter)":
        keep = nodes.nlargest(k, filter_by).account.tolist()
        H = G.subgraph(keep).copy()
        if hide_isolated:
            H.remove_nodes_from([n for n in list(H) if H.degree(n) == 0])
        H.add_nodes_from([] if hide_isolated else keep)
        caption = (f"The **{k} accounts with the highest {MEASURES[filter_by].split(' (')[0].lower()}**, "
                   f"and the transfers among them ({H.number_of_nodes()} accounts, {H.number_of_edges()} links shown).")
        return H, caption
    if view == "Ego network of one account":
        if acct not in G:
            return nx.DiGraph(), f"Account {acct} made no transfers."
        U = G.to_undirected(as_view=True)
        ego = list(nx.ego_graph(U, acct, radius=radius))
        if len(ego) > max_nodes:
            ego = [acct] + info.loc[[n for n in ego if n != acct], "degree"].nlargest(max_nodes - 1).index.tolist()
        H = G.subgraph(ego).copy()
        lab = "a laundering" if info.at[acct, "laundering"] else "a normal"
        caption = (f"Everyone within **{radius} hop(s)** of account **{acct}** ({lab} account) — "
                   f"{H.number_of_nodes()} accounts, {H.number_of_edges()} transfers-links.")
        return H, caption
    return None, ""


def pyvis_html(H):
    net = Network(height="680px", width="100%", directed=True, cdn_resources="in_line", bgcolor="#FFFFFF")
    vals = info.loc[list(H), size_by].astype(float)
    lo, hi = vals.min(), vals.max()
    for n in H.nodes():
        r = info.loc[n]
        z = 0.0 if hi == lo else (r[size_by] - lo) / (hi - lo)
        size = 8 + 32 * np.sqrt(z)                     # sqrt → AREA proportional to the value
        laund = bool(r.laundering)
        if colour_by == "AML label":
            col = C_LAUNDER if laund else C_NORMAL
        else:
            col = COMM_PAL[TOP_COMMS.index(r.community)] if r.community in TOP_COMMS else C_OTHER
        title = (f"Account {n}  ({'LAUNDERING' if laund else 'normal'})\n"
                 f"degree {r.degree}  (in {r.in_degree} / out {r.out_degree})\n"
                 f"money in ${r.money_in:,.0f} · out ${r.money_out:,.0f}\n"
                 f"PageRank {r.pagerank:.2e} · betweenness {r.betweenness:.4f}\n"
                 f"community {r.community}")
        net.add_node(int(n), label=str(n), size=float(size), color=col, title=title,
                     shape="diamond" if laund else "dot",     # redundant encoding: shape AND colour
                     font={"size": 10, "color": "#333"})
    for u, v, d in H.edges(data=True):
        both = info.at[u, "laundering"] and info.at[v, "laundering"]
        net.add_edge(int(u), int(v), width=float(0.5 + 1.2 * np.log10(1 + d["amount"] / 100)),
                     color="rgba(213,94,0,0.55)" if both else "rgba(150,150,150,0.35)",
                     title=f"{u} → {v}: ${d['amount']:,.0f} in {d['n_tx']} transfer(s)", arrows="to")

    if layout == "Force-directed":
        net.force_atlas_2based(gravity=-40, spring_length=90, damping=0.6)
    else:
        if layout == "Circular":
            order = sorted(H, key=lambda n: (info.at[n, "community"], n))
            pos = nx.circular_layout(order, scale=420)
        else:
            inner = [n for n in H if info.at[n, "laundering"]]
            outer = [n for n in H if not info.at[n, "laundering"]]
            pos = nx.shell_layout(H, nlist=[s for s in (inner, outer) if s], scale=420)
        for node in net.nodes:
            node["x"], node["y"] = map(float, pos[node["id"]])
            node["physics"] = False
        net.toggle_physics(False)
    return net.generate_html(notebook=False)


def community_figure():
    """Aggregated view: one bubble per Louvain community, width = money between communities."""
    cm = info.community
    flows = edges.assign(cs=edges.source.map(cm), ct=edges.target.map(cm))
    between = flows[flows.cs != flows.ct].groupby(["cs", "ct"]).amount.sum().reset_index()
    between["pair"] = between.apply(lambda r: tuple(sorted((r.cs, r.ct))), axis=1)
    pairs = between.groupby("pair").amount.sum()
    pairs = pairs[pairs >= min_flow * 1000]
    stats = nodes.groupby("community").agg(n=("account", "size"), share=("laundering", "mean"),
                                           measure=(size_by, "mean"))
    stats = stats[stats.n >= 20]
    A = nx.Graph()
    A.add_nodes_from(stats.index)
    for (a, b), w in pairs.items():
        if a in A and b in A:
            A.add_edge(a, b, weight=w)
    pos = nx.spring_layout(A, seed=42, weight="weight")
    fig = go.Figure()
    wmax = max([d["weight"] for *_, d in A.edges(data=True)] or [1])
    for a, b, d in A.edges(data=True):
        fig.add_trace(go.Scatter(x=[pos[a][0], pos[b][0]], y=[pos[a][1], pos[b][1]], mode="lines",
                                 line=dict(width=0.5 + 8 * d["weight"] / wmax, color="rgba(150,150,150,0.45)"),
                                 hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(
        x=[pos[c][0] for c in stats.index], y=[pos[c][1] for c in stats.index], mode="markers+text",
        text=[str(c) for c in stats.index], textposition="middle center", textfont=dict(color="#222", size=11),
        marker=dict(size=12 + 60 * np.sqrt(stats.n / stats.n.max()), color=stats.share * 100,
                    colorscale="Oranges", cmin=0, cmax=max(15, stats.share.max() * 100),
                    line=dict(color="#555", width=1),
                    colorbar=dict(title="% laundering<br>accounts")),
        customdata=np.c_[stats.n, stats.share * 100, stats.measure],
        hovertemplate="Community %{text}<br>%{customdata[0]} accounts<br>%{customdata[1]:.1f}% laundering"
                      f"<br>mean {size_by}: " + "%{customdata[2]:.3g}<extra></extra>",
        showlegend=False))
    fig.update_layout(height=640, xaxis=dict(visible=False), yaxis=dict(visible=False),
                      plot_bgcolor="white", margin=dict(l=10, r=10, t=30, b=10),
                      title=f"{len(stats)} communities (≥ 20 accounts) · size = # accounts · colour = % laundering · "
                            f"width = $ flowing between them")
    return fig, stats


# =====================================================================
#                               PAGE
# =====================================================================
st.title("🏦 Following the Money: an Anti-Money-Laundering Network Explorer")
st.caption("DATS 6401 · Week 6 homework — Graph & Network Data · Data: IBM AMLSim (Apache-2.0) synthetic bank transfers")

c = st.columns(5)
c[0].metric("Bank accounts (nodes)", f"{len(nodes):,}")
c[1].metric("Transfers", "120,558")
c[2].metric("Sender→receiver links (edges)", f"{len(edges):,}")
c[3].metric("Laundering accounts", f"{nodes.laundering.sum():,} ({BASE:.1%})")
c[4].metric("Modularity Q", f"{Q:.2f}", help=f"Louvain, {N_COMM} communities")

tab1, tab2, tab3, tab4 = st.tabs(["📖 1 · The network", "🕸️ 2 · Explore", "📊 3 · What structure reveals",
                                  "🧭 4 · Encodings, limitations & conclusions"])

# ---------------- TAB 1 ----------------
with tab1:
    l, r = st.columns([3, 2])
    with l:
        st.header("What this network represents")
        st.markdown(f"""
Banks must detect **money laundering**: moving illegal money through many accounts until its origin is hidden.
A single transfer looks innocent; the crime is visible only in **who sends money to whom** — a relational problem,
which is exactly what a network shows and a table hides.

**Data.** IBM's **AMLSim** simulator generates a realistic bank's transfer records for 150 days and plants known
laundering schemes inside them. I use its published sample `20K_fanin200cycle200`:

| Element | Meaning in the bank | Count |
|---|---|---|
| **Node** | a bank account | {len(nodes):,} |
| **Directed edge** u → v | account *u* sent money to *v* at least once | {len(edges):,} |
| **Edge weight** | total $ sent along that link (and # transfers) | — |
| **Node label** | account belongs to a laundering scheme (ground truth) | {nodes.laundering.sum():,} ({BASE:.1%}) |

**Two laundering typologies are planted** (200 of each):
- **Fan-in** — many accounts each send small amounts to one collector account (*structuring / smurfing*).
- **Cycle** — money moves A → B → C → … → A so the funds return to the owner looking "clean" (*round-tripping*).

**Graph construction.** 120,558 transfers were aggregated into one directed, weighted edge per sender→receiver
pair (15 self-transfers dropped). Measures were computed **once on the full graph** (then the app only filters):
degree, in/out-degree, money in/out, PageRank (α = 0.85), betweenness (Brandes, 500 sampled sources),
clustering, and Louvain communities on the undirected version.

> **Why synthetic?** Real bank transfer data is confidential by law. AMLSim is the standard public benchmark in AML
research precisely because it provides labelled laundering that real data never discloses.
""")
    with r:
        st.subheader("The question this app answers")
        st.info(f"**Can network structure alone point an investigator to laundering accounts?**\n\n"
                f"Short answer: **yes for hubs and loops, no for communities.** Among the 50 accounts with the most "
                f"counterparties, **{share_top50_deg:.0%}** are launderers vs **{BASE:.0%}** at random "
                f"(≈{share_top50_deg / BASE:.0f}× better). See tabs 3–4.")
        st.subheader("Laundering typologies")
        demo = nx.DiGraph()
        demo.add_edges_from([(f"s{i}", "C") for i in range(6)])
        demo.add_edges_from([("A", "B"), ("B", "D"), ("D", "E"), ("E", "A")])
        p = {f"s{i}": (np.cos(i * np.pi / 3) * 1 - 2.2, np.sin(i * np.pi / 3) * 1) for i in range(6)}
        p["C"] = (-2.2, 0)
        p.update({"A": (1.2, 0.8), "B": (2.4, 0.8), "D": (2.4, -0.8), "E": (1.2, -0.8)})
        fig = go.Figure()
        for u, v in demo.edges():
            fig.add_annotation(x=p[v][0], y=p[v][1], ax=p[u][0], ay=p[u][1], xref="x", yref="y", axref="x",
                               ayref="y", showarrow=True, arrowhead=3, arrowsize=1.3, arrowcolor=C_LAUNDER,
                               standoff=12, startstandoff=10)
        fig.add_trace(go.Scatter(x=[p[n][0] for n in demo], y=[p[n][1] for n in demo], mode="markers+text",
                                 text=["" if n.startswith("s") else n for n in demo], textfont=dict(color="white"),
                                 marker=dict(size=[34 if n == "C" else 18 if n.startswith("s") else 26 for n in demo],
                                             color=C_LAUNDER, symbol="diamond"), hoverinfo="skip"))
        fig.add_annotation(x=-2.2, y=-1.45, text="<b>Fan-in</b>: many → one collector", showarrow=False)
        fig.add_annotation(x=1.8, y=-1.45, text="<b>Cycle</b>: money returns home", showarrow=False)
        fig.update_layout(height=300, showlegend=False, xaxis=dict(visible=False), yaxis=dict(visible=False,
                          scaleanchor="x"), margin=dict(l=0, r=0, t=0, b=0), plot_bgcolor="white")
        st.plotly_chart(fig, width="stretch")
        st.caption("Fan-in creates a huge **in-degree** hub; a cycle creates closed loops (non-zero **clustering** "
                   "when accounts are reused). These are the structural fingerprints we look for.")

# ---------------- TAB 2 ----------------
with tab2:
    st.header("Explore the network")
    if view == "Communities (aggregated)":
        fig, stats = community_figure()
        st.markdown("**Aggregated view** — each bubble is a whole Louvain community collapsed into one node. "
                    "This is the *aggregate* escape from the hairball: readable at any size, but you can no longer "
                    "see individual accounts.")
        st.plotly_chart(fig, width="stretch")
        st.caption(f"Most communities sit near the {BASE:.0%} base rate (pale) — the darkest reaches only "
                   f"{stats.share.max():.0%}. Laundering is **spread across** communities, not concentrated in one.")
    else:
        H, caption = view_subgraph()
        st.markdown(caption)
        lg = st.columns(4)
        if colour_by == "AML label":
            lg[0].markdown(f"<span style='color:{C_LAUNDER};font-size:20px'>◆</span> **laundering account**",
                           unsafe_allow_html=True)
            lg[1].markdown(f"<span style='color:{C_NORMAL};font-size:20px'>●</span> **normal account**",
                           unsafe_allow_html=True)
        else:
            lg[0].markdown("**Colour = Louvain community** (8 largest coloured, rest grey) · ◆ = laundering",
                           unsafe_allow_html=True)
        lg[2].markdown(f"**Size** = {MEASURES[size_by]}")
        lg[3].markdown("<span style='color:#D55E00'>━</span> transfer between two launderers · "
                       "<span style='color:#999'>━</span> other", unsafe_allow_html=True)
        if H.number_of_nodes() == 0:
            st.warning("Nothing to draw with these settings — increase k or untick 'hide'.")
        else:
            show_html(pyvis_html(H), height=700)
            sub = info.loc[list(H)]
            m1, m2, m3 = st.columns(3)
            m1.metric("Accounts in view", len(sub))
            m2.metric("Laundering share in view", f"{sub.laundering.mean():.0%}",
                      f"{(sub.laundering.mean() - BASE) * 100:+.0f} pp vs {BASE:.0%} overall", delta_color="inverse")
            m3.metric("Money in view", f"${sub.money_in.sum():,.0f}")
            with st.expander("Table of the accounts in this view (sorted by the size measure)"):
                cols = ["laundering", "degree", "in_degree", "out_degree", "money_in", "money_out", "pagerank",
                        "betweenness", "clustering", "community"]
                st.dataframe(sub[cols].sort_values(size_by, ascending=False), width="stretch")
    st.caption("Tip: try *Top-k by in-degree* with *Shell* layout — the diamonds (launderers) move to the centre "
               "ring and you can count how many of the top accounts are suspicious.")

# ---------------- TAB 3 ----------------
with tab3:
    st.header("What the structure reveals")
    st.subheader("① Centrality as a red flag: % launderers among the top-k accounts")
    figL = px.line(LIFT, x="k", y="share", color="measure", markers=True, log_x=True,
                   color_discrete_sequence=["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00"],
                   labels={"share": "share of laundering accounts", "k": "top-k accounts (log scale)"})
    figL.add_hline(y=BASE, line_dash="dash", line_color="black",
                   annotation_text=f"random pick = {BASE:.1%}", annotation_position="bottom right")
    figL.update_yaxes(tickformat=".0%", range=[0, 1])
    figL.update_layout(height=420, plot_bgcolor="white", legend_title="ranked by")
    st.plotly_chart(figL, width="stretch")
    piv = LIFT.pivot(index="measure", columns="k", values="share")
    st.markdown(f"""
**Reading it:** if an investigator reviews the top 50 accounts by **degree**, **{piv.loc['degree', 50]:.0%}** are
launderers — about **{piv.loc['degree', 50] / BASE:.0f}×** the {BASE:.0%} base rate. **In-degree** and **money received**
work almost as well because the *fan-in* collectors receive from dozens of senders. **PageRank** is weaker
({piv.loc['pagerank', 50]:.0%}): it rewards being paid by important accounts, while launderers are paid by many *unimportant*
"smurf" accounts. All curves fall toward the base rate as k grows — centrality finds the **organisers**, not the
hundreds of small participants.
""")

    a, b = st.columns(2)
    with a:
        st.subheader("② Degree distribution by label")
        dd = nodes[nodes.degree > 0].groupby(["laundering", "degree"]).size().reset_index(name="count")
        dd["label"] = dd.laundering.map({0: "normal", 1: "laundering"})
        dd["share"] = dd["count"] / dd.groupby("label")["count"].transform("sum")
        figD = px.scatter(dd, x="degree", y="share", color="label", log_x=True, log_y=True,
                          color_discrete_map={"normal": C_NORMAL, "laundering": C_LAUNDER},
                          symbol="label", symbol_map={"normal": "circle", "laundering": "diamond"},
                          labels={"share": "share of accounts (log)", "degree": "degree (log)"})
        figD.update_layout(height=380, plot_bgcolor="white")
        st.plotly_chart(figD, width="stretch")
        med = nodes.groupby("laundering").degree.median()
        tail = nodes[nodes.degree > 100].laundering.value_counts()
        st.caption(f"Median degree: normal {med[0]:.0f}, laundering {med[1]:.0f}. In the long right tail "
                   f"(degree > 100) there are **{tail.get(1, 0)} laundering vs {tail.get(0, 0)} normal** accounts "
                   f"({tail.get(1, 0) / tail.sum():.0%} laundering vs {BASE:.0%} overall) — but legitimate hubs exist too.")
    with b:
        st.subheader("③ Closed loops (clustering > 0)")
        tri = pd.DataFrame({"label": ["normal", "laundering"], "share": [tri_n, tri_l]})
        figT = px.bar(tri, x="label", y="share", color="label", text_auto=".0%",
                      color_discrete_map={"normal": C_NORMAL, "laundering": C_LAUNDER})
        figT.update_yaxes(tickformat=".0%", range=[0, 1], title="accounts in ≥ 1 triangle")
        figT.update_layout(height=380, plot_bgcolor="white", showlegend=False)
        st.plotly_chart(figT, width="stretch")
        st.caption(f"**{tri_l:.0%}** of laundering accounts sit in at least one closed triangle vs **{tri_n:.0%}** of "
                   f"normal ones: cycles and reused mule accounts close loops that ordinary customers rarely form.")

    st.subheader("④ Communities do *not* isolate laundering")
    cs = nodes.groupby("community").agg(accounts=("account", "size"), share=("laundering", "mean")).reset_index()
    cs = cs[cs.accounts >= 100].sort_values("accounts", ascending=False)
    figC = px.bar(cs, x=cs.community.astype(str), y="share", text_auto=".0%",
                  color_discrete_sequence=[C_LAUNDER], labels={"x": "Louvain community (≥100 accounts, largest first)",
                                                               "share": "% laundering"})
    figC.add_hline(y=BASE, line_dash="dash", annotation_text=f"base {BASE:.1%}")
    figC.update_yaxes(tickformat=".0%")
    figC.update_layout(height=340, plot_bgcolor="white")
    st.plotly_chart(figC, width="stretch")
    st.caption(f"Modularity Q = {Q:.2f} (weak structure; ≥ 0.3 would be meaningful). In every large community "
               f"launderers make up {big_comm_share.min():.0%}–{big_comm_share.max():.0%}: the schemes are small "
               f"(a fan-in or a 5-account cycle) and get **absorbed** into big communities — the resolution limit "
               f"from the lecture, live.")

# ---------------- TAB 4 ----------------
with tab4:
    st.header("Why these encodings")
    st.markdown(f"""
| Visual channel | Encodes | Why this choice |
|---|---|---|
| **Position (x, y)** | layout algorithm (ForceAtlas2 / circular / shell) | the layout uses position, so data goes to other channels |
| **Node size** (area ∝ value) | a centrality you choose — default **degree** | size reads as "how much"; area (√ scaling) avoids exaggerating hubs 4× |
| **Node colour** | **AML label** (blue / vermillion) *or* Louvain community | hue reads as "which group"; Okabe–Ito palette is colour-blind safe |
| **Node shape** | ◆ laundering / ● normal | **redundant** with colour, so the label survives colour-blindness and grey-scale printing |
| **Edge arrow** | direction of money | in AML, *who pays whom* matters (fan-in ≠ fan-out) |
| **Edge width** | $ amount (log) | big flows stand out without one transfer dominating |
| **Edge colour** | orange = launderer → launderer | makes schemes visible inside a crowd |
| **Bubble colour (aggregated view)** | % laundering (sequential *Oranges*) | a quantity, so a light→dark scale, not hues |

**What they reveal.** With size = degree and colour = label, most of the biggest nodes in the top-k view are
diamonds: laundering collectors are the hubs of this network. In the ego view of a laundering hub (e.g. 9998: 172 senders,
201 receivers) you see a **star** — hundreds of counterparties around one account, the shape of a collector that
gathers money and then scatters it. Most of its neighbours are blue: the hub mixes dirty money with ordinary
customers' transfers, which is exactly how layering hides funds. Switching colour to
*community* shows that those stars sit inside ordinary communities, which is why community detection does not catch them.
""")
    st.header("Limitations — where this view breaks down or could mislead")
    st.warning(f"""
1. **Hairball / filtering bias.** 20,000 accounts cannot be drawn, so every view is a filter. The top-k view is
   *chosen by centrality*, so it will always look full of hubs; the many launderers who are small
   participants (e.g. the senders in a fan-in) are mostly **invisible** in it. The shown laundering share is for the
   view, not the bank.
2. **Position means nothing numerically.** Force-directed coordinates change with each run and drag; two accounts
   that look far apart can be one transfer apart. Only *grouping* is meaningful, never distance or direction on screen.
3. **Communities are a heuristic.** Louvain (seed 42) gives {N_COMM} communities with Q = {Q:.2f}; another seed or
   algorithm (Infomap) gives a different split. Community colours are arbitrary labels, and only the 8 largest get a hue.
4. **Betweenness is approximate** (500 sampled sources) and treats every link as equal length, ignoring $ amounts.
5. **Synthetic data.** AMLSim plants textbook typologies; real launderers adapt (layering through other banks,
   cash, crypto) and real normal customers include legitimate hubs — payroll accounts, merchants, utilities — that
   would also have huge in-degree. **High centrality is a reason to look, not proof of guilt**; using it as an
   automatic accusation would produce many false positives.
6. **Time is collapsed.** Five months of transfers become one static edge; the *order* of transfers (money arriving
   then leaving within hours) is a key AML signal this view cannot show.
""")
    st.header("Conclusions")
    st.success(f"""
- **Network structure finds organisers.** Ranking by degree, in-degree or money received puts
  **{min(piv.loc[['degree', 'in_degree', 'money_in'], 50]):.0%}–{max(piv.loc[['degree', 'in_degree', 'money_in'], 50]):.0%}** launderers in the top 50, versus {BASE:.0%} by chance
  (≈{share_top50_deg / BASE:.0f}× lift) — a table of individual transfers would never show this.
- **The right centrality depends on the question.** Fan-in schemes are caught by **in-degree**; PageRank, which
  rewards *important* payers, is weaker because launderers are paid by many *unimportant* mule accounts.
- **Loops are a fingerprint.** {tri_l:.0%} of laundering accounts are in a closed triangle vs {tri_n:.0%} of normal accounts.
- **Communities are the wrong tool here** (Q = {Q:.2f}): schemes are too small and get absorbed into large
  communities. Pattern-level measures (in-degree, clustering, cycles) beat group-level ones for AML.
- **Practical use:** an analyst would use the top-k and ego views as a **triage list** for investigation,
  then confirm with transaction timing and customer information.
""")
    st.caption("Built with NetworkX · PyVis · Plotly · Streamlit. Data: IBM AMLSim sample 20K_fanin200cycle200 "
               "(github.com/IBM/AMLSim, Apache-2.0).")
