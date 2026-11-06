# Phase 2 roadmap: drift detection & automated retraining (not built yet)

This describes the intended design for the next phase of this repo — an
**evaluation pipeline** that watches a deployed model over time, detects
data drift / skew / performance degradation, and automatically re-triggers
training with new data. It is documented here so the phase-1 architecture
(this repo, today) is built with it in mind, without actually implementing
it yet. Build this once you're comfortable with everything in phase 1.

## Why phase 1 is already set up for this

- **`predictions` table** (written by each project's FastAPI `/predict`
  endpoint) already stores every request's input features, the prediction,
  the **model version** that produced it, and a timestamp. This is exactly
  the data an evaluation pipeline needs: a time-ordered log of what the
  model saw and said in production.
- **`data_generation.py`** in each project is a parameterized function, not
  a one-off script — it takes explicit arguments for the underlying
  generating process (e.g. pricing coefficients, category frequencies,
  noise level). Phase 2's synthetic "data shift injector" can call the same
  function with *different* parameters per batch to simulate real-world
  change, rather than needing a new data source.
- **MLflow model registry** already tracks every trained version and its
  metrics, so "new reference model" is always available for comparison.

## Planned components

1. **Synthetic drift injector** — extends `data_generation.py` to produce
   successive batches where the generating distribution shifts over
   "time" (e.g. average square footage trends up, a `neighborhood`
   category grows in frequency, the price-per-sqft coefficient changes).
   This stands in for the real world changing, without needing to wait for
   it or find a real dataset that happens to drift on request.

2. **`evaluation` Airflow DAG** (new, alongside the existing `training_dag`)
   - Pull the **reference window**: the data the current `Production`
     model was trained on.
   - Pull the **current window**: recent rows from the `predictions`
     table (plus, once available, actual outcomes joined back in — e.g.
     the real sale price, or whether the customer actually churned).
   - Compute **data drift** (are current input distributions different
     from the reference — e.g. via population stability index or a
     library like `evidently`), and, once actuals are available, **live
     performance metrics** (same metrics as phase 1's `evaluate.py`,
     computed on production data instead of the held-out test set).
   - Compare against thresholds (e.g. "RMSE degraded by more than 15%",
     "PSI above 0.2 on more than N features").

3. **Auto-retrain trigger** — if a threshold is crossed, the `evaluation`
   DAG triggers the existing `training_dag` (via Airflow's
   `TriggerDagRunOperator`, or by calling the same FastAPI `/train`
   endpoint phase 1 already exposes) — so retraining reuses the exact
   pipeline already built, not a new one.

4. **Actuals feedback loop** — a mechanism to eventually learn the true
   outcome for a past prediction (e.g. the house actually sold, the
   customer's contract actually ended) and write it back, so live
   performance metrics — not just data drift — can be computed. In a real
   system this comes from downstream business events; here it can be
   simulated by generating "ground truth" alongside each synthetic batch
   and joining on a delay.

## Open questions to resolve when you start phase 2

- Where does phase 2 live: a third top-level project, or a module added to
  each existing project? (Leaning toward a third project that *depends on*
  both, since the evaluation logic is nearly identical for regression and
  classification and shouldn't be duplicated — decide once phase 1 is
  solid and you've seen how much of `evaluate.py` naturally generalizes.)
- What drift-detection library to standardize on (`evidently` is the most
  common local/open-source choice as of writing).
- What counts as "enough" reference data / production data before a drift
  computation is statistically meaningful, rather than noise.
