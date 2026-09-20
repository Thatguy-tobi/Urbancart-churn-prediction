# UrbanCart — Churn Early Warning

Predicting which UrbanCart customers are about to stop ordering, early enough for the
retention team to act.

**Milestone 2 of the UrbanCart case study.** UrbanCart is a mid-size e-commerce retailer
selling home goods and electronics. Each milestone is a separate request from a different
internal team; the client never specifies a technique, so choosing one and justifying it
is part of the work.

---

## The brief

Retention's words: *customers quietly stop ordering, and we don't notice until the
monthly report — by then they're long gone.* They want something that flags it earlier,
using data already tracked on every account.

## The data

`data/milestone-2-customer-churn.csv` — 20,000 customers. No missing values, no
duplicates.

| Column | Description |
|---|---|
| `MonthsActive` | Months as a customer |
| `AvgOrderValue` | Average order size |
| `NumOrdersLastQuarter` | Orders in the last three months |
| `DaysSinceLastOrder` | Days since their most recent order |
| `SupportTicketsFiled` | Support tickets raised |
| `Churned` | **Target** — 1 if they stopped ordering, 0 otherwise |

## How I framed it

`Churned` is a yes/no flag, so this is **supervised classification**.

The complication that shapes everything: only **7.8% of customers churned** — 1,567 of
20,000, roughly 1 churner for every 12 retained customers. That imbalance decides which
metrics are honest and which are actively misleading.

## Why accuracy would be the wrong yardstick

A `DummyClassifier` that predicts "nobody churns" — without looking at the data at all —
scores **92.2% accuracy** and catches **0 of 313** churners in the test set.

It gets worse. At the default 0.50 threshold, both Logistic Regression and Gradient
Boosting score **92.7%** — *better than the do-nothing baseline* — while each catching
only 33 churners. Grading on accuracy would have ranked the two least useful models
highest and sent retention away with a model that finds almost nobody.

Metrics used instead:

| Metric | What it asks |
|---|---|
| **Recall** | Of the customers who really left, how many did we flag? |
| **Precision** | Of the customers we flagged, how many really were leaving? |
| **F1 / F2** | Both combined; F2 weights recall ~4× higher |
| **PR-AUC** | Performance on the rare class across all thresholds — random guessing scores 0.078 here, not 0.5 |

Accuracy is still reported in the notebook, purely to show how little it separates the
models.

## Which mistake costs more

A **false negative** is a churner we failed to flag: the customer leaves silently, we
lose their entire future value, and there is no second chance — this is precisely the
failure retention reported. A **false positive** is one unnecessary win-back email.

These are not close, so recall outweighs precision here, and **F2** is the selection
metric.

## Method

1. **Stratified 80/20 split** before any training, preserving the 7.8% churn rate in
   both halves.
2. **Preprocessing inside a `Pipeline`** so the scaler fits on training folds only.
3. **Four models** including the dummy baseline, with `class_weight="balanced"` where
   supported.
4. **Decision threshold tuned on out-of-fold training predictions only.** Tuning it on
   the test set would be leakage and the reported scores would stop being an estimate of
   unseen performance.
5. **Test set scored once**, at the end.

## Results

Held-out test set, 4,000 customers, 313 of them real churners.

| Model | Threshold | Precision | Recall | F1 | F2 | PR-AUC |
|---|---|---|---|---|---|---|
| **Logistic Regression** | 0.07 | 0.181 | **0.674** | 0.285 | **0.436** | **0.353** |
| Gradient Boosting | 0.09 | 0.204 | 0.559 | 0.299 | 0.414 | 0.334 |
| Random Forest | 0.08 | 0.164 | 0.610 | 0.259 | 0.395 | 0.323 |

Every tuned threshold landed near the base rate (0.07–0.09) rather than 0.50 — with
churn this rare, almost no customer ever reaches a 50% predicted probability, which is
exactly why the untuned models caught so few.

**Confusion matrix — Logistic Regression @ 0.07**

|  | Actually churned | Actually stayed |
|---|---|---|
| **Predicted churn** | TP = 211 | FP = 955 |
| **Predicted stay** | FN = 102 | TN = 2,732 |

- Flags **1,166 of 4,000** customers (29%) for outreach
- Catches **211 of 313** real churners — **67%**
- Precision 0.181 against 0.078 at random: a **2.3× lift**
- **5.5 contacts per churner reached** — the figure a campaign can be planned around
- Accuracy falls to 0.74, *below* the do-nothing baseline. That is the correct trade.

**What drives risk:** `DaysSinceLastOrder` dominates — churners averaged 42 days since
ordering against 18 for everyone else — with `SupportTicketsFiled` second. Notably,
`AvgOrderValue` barely differs between the groups: UrbanCart is **not** losing customers
because they were low-value. It is losing quiet, frustrated ones of every size.

## Recommendation

**Logistic Regression, with the decision threshold set to ~0.07 rather than 0.50.**

- Leads on both metrics that matter here — F2 0.436 and PR-AUC 0.353.
- Outputs a **calibrated probability**, so retention can sort a call list worst-first and
  work down it as capacity allows, rather than receiving an undifferentiated yes/no.
- Stays **explainable**: "42 days quiet, three support tickets" is a reason a human can
  check and act on.
- The **threshold is a dial the business owns** — lower it to catch more, raise it for a
  shorter, higher-precision list. No retraining needed.

**Tradeoffs accepted:**

1. Roughly 4 in 5 flagged customers were not leaving. That is the deliberate price of
   67% recall, and it is only the right price while outreach stays a cheap email. If it
   becomes phone calls or real discounts, raise the threshold and accept lower recall.
2. 1 in 3 churners is still missed. This is early warning, not a guarantee.
3. Five features is a thin picture. Browsing activity, email engagement, and support
   ticket *sentiment* rather than a raw count are the obvious next additions.

**Measurement warning for whoever operates this.** Once retention starts intervening,
flagged customers who then stay will look like false positives — but some stayed
*because* of the intervention. Model performance and campaign performance stop being
separable. Hold back a small random untreated control group, or a model that is working
will look like one that is failing.

---

## Repository structure

```
urbancart-churn-prediction/
├── data/
│   └── milestone-2-customer-churn.csv
├── notebooks/
│   ├── UrbanCart_Milestone2.ipynb            # full analysis, outputs included
│   └── UrbanCart_Milestone2_Explained.ipynb  # identical results, plain-English commentary
├── src/
│   └── milestone2_customer_churn.py          # the same pipeline as a runnable script
├── reports/
│   └── UrbanCart Milestone 2.pptx            # 11-slide summary for the client
├── requirements.txt
├── .gitignore
└── README.md
```

## Running it

```bash
pip install -r requirements.txt
jupyter notebook notebooks/UrbanCart_Milestone2.ipynb
```

Or run the pipeline directly:

```bash
python src/milestone2_customer_churn.py
```

The notebooks look for the CSV at `data/milestone-2-customer-churn.csv` first, then in
their own folder, so the structure above works unmodified. `random_state=42` throughout,
so every figure in this README reproduces exactly.

## A note on the two notebooks

`UrbanCart_Milestone2_Explained.ipynb` contains **identical code and identical results**
to `UrbanCart_Milestone2.ipynb`. Only the commentary differs — it is rewritten for
readers without a machine learning background. Either one can be run; neither depends on
the other.
