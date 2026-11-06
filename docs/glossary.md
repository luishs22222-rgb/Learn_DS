# Glossary: core Data Science / ML concepts

This is the shared reference for both projects in this repo
(`regression-house-prices/` and `classification-churn/`). Each project's
README links back here instead of re-explaining these ideas twice. Skim it
once end to end, then use it as a lookup while you work through the code.

Concepts are grouped in roughly the order you'll meet them in the pipeline:
**data → features → training → evaluation → engineering/serving → the
future (monitoring)**.

---

## 1. Data

**Relational database (mock)**
A database made of multiple related tables (e.g. `properties`, `locations`,
`sales`) connected by keys, queried with SQL — as opposed to one flat CSV.
Real companies almost never hand you a clean single table; you write SQL
(often with `JOIN`s and `GROUP BY`) to assemble the dataset you actually
train on. Both projects use a local SQLite file to stand in for "the
production database" — same SQL, same `pandas.read_sql`, just no server to
run.

**EDA — Exploratory Data Analysis**
The first thing you do with any new dataset, *before* modeling: look at
distributions, ranges, missing values, correlations, obvious outliers or
data-entry errors. The goal is to build intuition ("does this data make
sense?") and to spot problems (nulls, skew, leakage) early. Done with
`pandas.describe()`, `.info()`, histograms, scatter plots, correlation
heatmaps (`matplotlib`/`seaborn`).

**Feature**
A single input column used by the model — e.g. `sqft`, `bedrooms`,
`neighborhood`. Feature *engineering* is the work of turning raw columns
into inputs a model can actually learn from (below).

**Missing values (nulls) & imputation**
Real data has gaps (a sensor didn't report, a field was optional). Models
generally can't accept `NaN` directly. You choose a strategy: drop the row,
drop the column, or *impute* — fill with a statistic (median for numeric,
most-frequent/constant for categorical). We deliberately inject some real
nulls into the synthetic data so you have to handle this for real, not just
read about it.

**Categorical encoding (one-hot encoding)**
Models need numbers, not strings. One-hot encoding turns a categorical
column like `contract_type ∈ {month-to-month, one-year, two-year}` into
several binary (0/1) columns, one per category. This avoids implying a fake
order or magnitude between categories (unlike just mapping them to
`0, 1, 2`, which would tell the model "two-year" is "twice" "one-year").

**Scaling / normalization**
Rescaling numeric features (e.g. to mean 0, std 1 via `StandardScaler`) so
features with big raw ranges (like `total_charges` in the hundreds) don't
dominate features with small ranges (like `bedrooms` in single digits).
Linear/logistic regression and distance-based models need this; tree-based
models (random forests) are scale-invariant and don't.

**Train / test split**
You never evaluate a model on the same data it learned from — that tells
you how well it *memorized*, not how well it *generalizes*. So you split
the dataset (typically 70-80% train / 20-30% test) before touching the
model. For classification, the split is *stratified* (each class kept in
the same proportion in both halves) so a rare class doesn't disappear from
one side. Later, when data changes over time, a *chronological* split
(train on the past, test on the future) becomes more realistic than a
random split — noted here because phase 2 will care about this.

---

## 2. Modeling

**Model**
A function, fit to data, that maps inputs (features) to an output
(prediction). "Training" means finding the internal numbers (parameters)
that make that function fit the training data well, according to some loss.

**Regression vs. classification**
Two families of supervised learning problems. *Regression* predicts a
continuous number (house price in dollars). *Classification* predicts a
category (will this customer churn: yes/no). They use different models,
different loss functions, and — importantly — different, less intuitive
evaluation metrics for classification (see below), which is exactly why
project 2 is harder to interpret than project 1.

**Parameters vs. hyperparameters**
*Parameters* are learned automatically during training (e.g. the
coefficients in a linear regression). *Hyperparameters* are choices you
make before training that control how learning happens (e.g. number of
trees in a random forest, `max_depth`). We keep hyperparameters simple and
explicit in this repo rather than doing a full search, to keep focus on the
pipeline, not tuning.

**Loss function vs. evaluation metric**
Easy to conflate — they are not the same thing. The *loss* is what the
optimizer directly minimizes during training (e.g. mean squared error for
linear regression, log-loss for logistic regression) — it must be
differentiable/optimizable. The *metric* is what a human reads afterward to
judge quality (MAE, R², precision, ROC-AUC...) — it's chosen for
interpretability, not optimizability. A model can be trained on one loss
and judged on several different metrics.

**scikit-learn `Pipeline` / `ColumnTransformer`**
A `ColumnTransformer` bundles per-column preprocessing (impute this column,
one-hot that one, scale these) into one object. A `Pipeline` chains that
preprocessing with the model itself into a *single* object with `.fit()`
and `.predict()`. Why this matters: whatever transformation you applied at
training time is applied *identically* at prediction time, because it's
the same fitted object — this eliminates "training/serving skew," a very
common real-world bug where production preprocessing quietly drifts from
what the model was trained on.

**Reproducibility (random seeds)**
Data generation, train/test splitting, and some models (random forests)
involve randomness. We fix a seed everywhere it matters so runs are
repeatable — a prerequisite for meaningfully comparing "new model vs.
previous model."

### Regression metrics (project 1)
- **MAE (Mean Absolute Error)** — average absolute $ difference between
  prediction and truth. Same units as the target, easy to explain to
  anyone ("off by $18k on average").
- **RMSE (Root Mean Squared Error)** — like MAE but squares errors first,
  so big misses are penalized disproportionately. Also in target units.
- **R² (coefficient of determination)** — fraction of the variance in the
  target explained by the model, 0 to 1 (can go negative for a bad model).
  Useful for comparing models, less intuitive on its own than MAE/RMSE.

### Classification metrics (project 2 — the harder half)
- **Accuracy** — fraction of predictions correct. *Misleading under class
  imbalance*: if only 5% of customers churn, a model that always predicts
  "no churn" gets 95% accuracy while being useless.
- **Confusion matrix** — a 2×2 (for binary) table of true positives, false
  positives, true negatives, false negatives. The source of truth every
  other classification metric is computed from.
- **Precision** — of everything the model flagged positive, what fraction
  actually was. High precision = few false alarms.
- **Recall** — of everything that was actually positive, what fraction the
  model caught. High recall = few missed cases.
- **Precision/recall tradeoff** — you can usually push one up by pushing
  the other down (by moving the decision threshold). Which one matters more
  depends on the cost of each error type (e.g. missing a churner vs.
  wasting a retention offer on someone who wasn't leaving).
- **F1 score** — harmonic mean of precision and recall, one number that
  balances both.
- **ROC-AUC** — how well the model ranks positives above negatives across
  *all* possible thresholds, 0.5 (random) to 1.0 (perfect), independent of
  where you set the decision cutoff.

The classification project's README expands each of these with worked
numbers from that project's actual model.

---

## 3. MLOps / engineering

**Experiment tracking**
Every training run logs its parameters, metrics, and the resulting model
artifact somewhere queryable, instead of scattered print statements or
spreadsheets. We use **MLflow** for this (local, no cloud).

**Model registry & promotion**
A versioned catalog of trained models with lifecycle stages (e.g. `None` →
`Staging` → `Production`). "Promotion" means: after training a new
candidate, evaluate it on the same held-out test data as the current
`Production` model and only replace `Production` if the new one is
actually better — never blindly overwrite a working model. This is the
concrete mechanism behind "compare new model vs. previous, promote if
better."

**Orchestration / DAG (Directed Acyclic Graph)**
A pipeline expressed as a graph of tasks with dependencies (generate data →
validate → build features → train → evaluate → promote), run and scheduled
by an orchestrator — **Apache Airflow** here. Compared to one long script,
a DAG gives you per-task retries, logging, scheduling, and a UI showing
what ran, when, and whether it failed — necessary once a pipeline needs to
run repeatedly and reliably rather than once by hand.

**Data validation**
Cheap sanity checks run on data *before* it's trusted for training — row
counts, null-rate thresholds, expected columns/types. Catches upstream
data problems before they silently degrade a model. We implement this
here as a plain pandas assertion step; production systems often use a
dedicated library (`pandera`, `great_expectations`) — worth knowing exists,
not required to learn the concept.

**Batch vs. online serving**
*Online* (a.k.a. synchronous/real-time) serving answers one prediction
request at a time over an API — what our FastAPI `/predict` endpoint does.
*Batch* serving scores a large file/table of inputs all at once on a
schedule. Same trained model, different delivery mechanism; which one you
need depends on whether a human/system is waiting on the answer right now.

---

## 4. Looking ahead (not built yet)

**Data drift / skew**
*Data drift* — the distribution of input features changes over time (e.g.
average `sqft` creeps up, a new `neighborhood` appears) even though the
relationship to the target hasn't. *Concept drift* — the relationship
between features and target itself changes (the same house is now worth
more/less for reasons the old model never saw). *Training/serving skew* —
a related but different problem where production data is processed
differently than training data (which our shared `Pipeline` objects
prevent, see above). All three can silently erode a "working" model.

**Performance degradation monitoring**
Comparing a live model's real-world metrics (once you can join predictions
back to actual outcomes) against its training-time metrics, to detect when
it's time to retrain — the trigger condition for an automated retraining
pipeline. See `docs/roadmap-phase2.md` for how this repo is set up to add
that later.
