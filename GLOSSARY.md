# sage-xgb

Detects money-laundering transactions by feeding graph neural network output to a gradient-boosted classifier.

## Language

### Data

**AMLworld**:
The IBM synthetic anti-money-laundering dataset (Altman et al., 2023), published on Kaggle as "IBM Transactions for AML". Each Variant ships a transactions CSV, an accounts CSV and a Patterns file.
_Avoid_: AMLSim (for the dataset), IBM AML dataset

**AMLSim**:
IBM's older Java transaction simulator, a different generator from the one behind AMLworld. This project does not use it.

**Variant**:
One of the six AMLworld datasets, named by illicit ratio (HI or LI, for high or low) and size (Small, Medium or Large). The first milestone covers HI-Small only.
_Avoid_: split, version, subset

**Transaction**:
One row of the transactions CSV: a directed money transfer between two Accounts at a minute-resolution timestamp, carrying the label `Is Laundering`. It is the unit that is scored and the edge of the graph.
_Avoid_: payment, row, record

**Account**:
A node of the graph, identified by the pair (bank, account number). The account number alone is not unique across banks (HI-Small has 515,080 distinct numbers but 515,088 distinct pairs).
_Avoid_: wallet, holder, entity

**Illicit transaction**:
A Transaction with `Is Laundering = 1`. HI-Small has 5,177 of 5,078,345 (0.102%).
_Avoid_: fraud, suspicious transaction, positive (outside metric definitions)

**Pattern**:
One labelled laundering attempt in the Patterns file, of one of eight typologies (fan-out, fan-in, gather-scatter, scatter-gather, cycle, bipartite, stack, random), listing the Transactions that make it up. HI-Small has 370. Patterns are for evaluation and analysis only, never a feature source.
_Avoid_: ring, typology instance

**Tail**:
The Transactions dated after the Variant's nominal window. In HI-Small the window is 2022-09-01 to 2022-09-10, and the Tail is 1,108 Transactions on 2022-09-11 to 2022-09-18, of which 655 are illicit.
_Avoid_: overflow, post-window rows

### Training

**Window**:
A trailing period `w` (1 hour, 24 hours or 7 days) before a Transaction at time `t`, the half-open interval [t - w, t) on minute timestamps. Transactions in the same minute are not in each other's window.
_Avoid_: lookback, rolling period

**Day block**:
A run of consecutive training days that the GNN cross-fit treats as one unit. HI-Small uses days 1-2, 3-4 and 5-6.
_Avoid_: fold (alone), chunk

**Snapshot**:
The graph of Transactions from the days before day d, used to embed Accounts for Transactions of day d.
_Avoid_: graph state, cache

**Out-of-fold score**:
A GNN score for a Transaction produced by a GNN that never trained on that Transaction's day or any later day.
_Avoid_: held-out prediction
