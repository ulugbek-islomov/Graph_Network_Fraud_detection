# 🏦 AML Network Explorer — DATS 6401 Week 6 Homework

An interactive Streamlit app that explores a **bank-transfer network** to see whether network structure can point
an investigator to **money-laundering** accounts.

## Run it

```bash
pip install -r requirements.txt
streamlit run app.py
```

The processed data is included in `data/`. To rebuild it from the original source, run `python prepare_data.py`. That
script downloads the IBM AMLSim sample from GitHub and computes all the measures (takes about 2 minutes). The app
also runs it automatically if the `data/` folder is empty.

## Files

| File | Purpose |
|---|---|
| `app.py` | the Streamlit app (all text, explanations and conclusions are inside the app) |
| `prepare_data.py` | download script + graph construction + structural measures |
| `data/nodes.csv` | 20,000 accounts with degree, in/out-degree, money in/out, PageRank, betweenness, clustering, Louvain community, laundering label |
| `data/edges.csv` | 117,326 directed sender→receiver links with total amount and number of transfers |
| `requirements.txt` | packages |

## Data

**IBM AMLSim** (Apache-2.0), sample `20K_fanin200cycle200` — <https://github.com/IBM/AMLSim>.
It contains 20,000 synthetic bank accounts, 120,558 transfers over 150 days, and 200 fan-in plus 200 cycle laundering
patterns. 1,804 accounts (9.0%) are labelled as part of a laundering scheme. Real bank transfer data is
confidential, so AMLSim is the standard public benchmark for anti-money-laundering (AML) research.

## How the app meets the rubric

| Rubric item | Where in the app |
|---|---|
| **Graph construction + structural measure (3)** | Transfers aggregated into a directed, weighted graph (`prepare_data.py`). Measures: degree, in/out-degree, PageRank, betweenness (Brandes, k = 500), clustering, Louvain communities and modularity. Tab 1 explains the construction; Tab 3 analyses the measures. |
| **Effective, accessible encodings (3)** | Size = chosen centrality (area-scaled). Colour = AML label or community (Okabe–Ito colour-blind-safe palette). Shape = ◆ laundering / ● normal (redundant with colour). Arrows = money direction; edge width = $ amount; orange edges = launderer → launderer. Aggregated view: sequential colour for % laundering. |
| **Meaningful interactive control (2)** | Sidebar: 3 views (top-k centrality filter / ego network / aggregated communities), filter measure + k slider, size measure, colour mode, 3 layouts, ego account + radius + node cap, flow threshold. PyVis graph: drag, hover, zoom. |
| **Justification + limitations (2)** | Tab 4: encoding table with reasons, what the encodings reveal, 6 limitations, conclusions. |

## Key findings

- Among the top 50 accounts by degree, **76%** are launderers, against a **9%** base rate (about 8×).
- **81%** of laundering accounts sit in a closed triangle, against **12%** of normal accounts.
- Louvain communities (Q = 0.24) do **not** isolate laundering: every large community has 6–14% launderers.
  Small schemes get absorbed into large communities (the resolution limit).
