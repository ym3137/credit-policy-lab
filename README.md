# Credit Policy Lab

**At a fixed approval rate, which screening policy produces the lowest bad rate
— and what does tightening cost the customers who would have repaid?**

This is a policy study, not a default-prediction project. The model is an
intermediate step; the deliverable is the trade-off curve a credit team uses to
set a cut-off, plus an honest account of how long that cut-off stays valid.

Built on LendingClub's public 2007–2018 loan data. Every number here comes from
an out-of-time test set that is touched exactly once.

---

## The three questions this answers

**1. Where do you set the approval rate?**
Loosen and you grow but lose more; tighten and you are safe but small. That is
a business decision, not a modelling one. The model's job is to draw the
trade-off curve so the business can pick a point on it.

**2. How does a score become a policy?**
A probability cannot be executed. It has to become *approve above this cut-off,
decline below that one, refer the middle band to manual review.* Choosing those
boundaries, and deciding whether they should differ by segment, is the work.

**3. When does the policy stop working?**
Populations drift. A model that ranked well at launch may not six quarters
later. This repo measures that decay across eleven years of vintages and names
the metric that should trigger a retrain.

## Verified against the real data

The pipeline has been run end-to-end on the full file (2,260,701 loans).
Full detail in [`docs/audit_findings.md`](docs/audit_findings.md); the three
numbers that matter:

**40.36% of the raw file is censored** — 878,317 loans still `Current`, plus
grace-period and late states. Mapping `Current` to the good class, which is the
usual shortcut, would mislabel 878k loans. After the term filter, issue window
and status resolution, the modelling sample is **589,488 loans at a 14.03% bad
rate** — 26% of the raw file.

**The leakage alarm fired on 9 fields; all 9 were post-origination.**
`last_fico_range_high` scores IV **3.15** against the default target — that is
not a predictor, it is the answer key, and it sits one underscore away from the
legitimate `fico_range_high` in the same file.

**No single application-time feature is strong.** The best legitimate one is
FICO at IV 0.125; everything else is weak. LendingClub's own `sub_grade` scores
0.305, because it is already a composite of those same inputs. So the platform
benchmark is a real opponent, not a straw man — if the scorecard merely matches
it, that is the honest result, and a project claiming to crush the incumbent on
this data should be distrusted.

**The platform benchmark wins standalone, and the stack beats it.** On the
test set — evaluated once — neither the scorecard (AUC 0.666) nor gradient
boosting (0.669) beats LendingClub's own `sub_grade` (0.681) from
application-time data alone. The platform had inputs this data does not expose.
But stacking the scorecard *on top of* the grade beats the grade alone by
**30–37 basis points of approved bad rate at every approval rate from 50% to
80%**, and the sign held from validation through to test. Incremental lift over
the incumbent is also the question a risk team actually asks.

**Calibration drifted even though the score distribution did not.** Score PSI
train→test is 0.0023 — stable — while predicted bad rate runs 1.5 to 2.4 points
below observed across deciles, because train bad rate is 13.25% and test is
14.71%. PSI monitoring alone would have reported no problem. A policy set on a
probability cut-off would have silently under-delivered; one set on an approval
*rate* is immune. That is a concrete argument for how to express credit policy,
and only out-of-time evaluation surfaces it.

The target is also **non-stationary**: bad rate runs 12.3% (2013) to 14.9%
(2015), with test ~1.5 points above train. Part of any out-of-time decay is the
population moving rather than the model degrading, and the decay analysis has
to separate the two.

## What the pipeline does

| Module | Output |
|---|---|
| `00_data_audit` | Field eligibility split, fill-rate drift by vintage, leakage alarm |
| `01_labeling_and_splits` | Three-way label mapping, performance window, out-of-time splits, vintage curve |
| `02_scorecard` | WOE bins, IV screening, logistic scorecard, auditable points table |
| `03_model_comparison` | Simple rule vs scorecard vs gradient boosting |
| `04_policy_simulation` | **Trade-off curves, platform-grade benchmark, rough economics** |
| `05_stability_monitoring` | KS decay by cohort, PSI drift, calibration, retrain signal |
| `06_reject_inference` | Approved-vs-declined population shift, sensitivity range |

### Three design decisions worth reading before the code

**Censored loans are dropped, not labelled good.** A loan still `Current` has
not had the chance to default. Mapping it to 0 imports future defaults into the
good class and inflates every metric downstream. See `docs/methodology.md`.

**The platform's own grade is held out of the feature set.** `grade`,
`sub_grade` and `int_rate` are known at origination, so they are not leakage —
but they are LendingClub's model output, not applicant attributes. Using them
would mean learning to imitate the platform's scorecard. They are reserved as
the benchmark instead: *at the same approval rate, does this policy beat the
platform's own ordering?* See `docs/leakage_fields.md`.

**High IV is treated as a bug report.** Roughly a third of the 151 columns are
generated after money goes out. Leaving them in yields AUC above 0.95 and a
model that cannot be deployed. `src/leakage.py` flags any feature with IV above
0.50 — against a 36-month default target, that is almost always an excluded
field that slipped through, not a discovery.

## What this does **not** show

Stated plainly, because these are the first things a reviewer should ask about:

- **No causal claim.** This compares screening policies. It does not show that
  declining an applicant, or contacting them, prevents a default.
- **Selection on approval.** Only funded loans have outcomes, so every model is
  conditional on passing LendingClub's own credit policy. Reject inference is
  attempted but **cannot be validated** — declined applicants have no
  counterfactual outcome, ever. `06_` reports a sensitivity range, not a fix.
- **No recession in the data.** The 2012–2015 modelling window is entirely
  expansion-era. Nothing here demonstrates behaviour under stress.
- **Economics are illustrative.** The return module uses stated rates and an
  assumed flat recovery rate, with no funding cost, servicing cost or
  collections model. It ranks policies against each other on shared
  assumptions; it does not estimate profit.
- **US marketplace lending, 36-month term only.** Not transferable to
  bank-originated credit or other markets without re-estimation.

## Reproducing

```bash
pip install -r requirements.txt

# Download from Kaggle: wordsforthewise/lending-club
# Place both CSVs in data/raw/ (paths configured in config/config.yaml)

python tests/test_core.py          # synthetic smoke test, no raw data needed
jupyter lab notebooks/             # then run 00 → 06 in order
```

All windows, thresholds and label mappings live in `config/config.yaml`.
Nothing is hard-coded in a notebook.

## Repository layout

```
config/config.yaml      windows, splits, label mapping, thresholds
docs/
  methodology.md        labels, performance window, splits, limitations
  leakage_fields.md     three-way field eligibility + the IV alarm
  data_dictionary.md    business meaning of every column used
src/
  labeling.py           three-way target mapping, performance window
  splits.py             out-of-time splits (no random split exists here)
  binning.py            WOE / IV, monotonicity, leakage screening
  scorecard.py          logistic scorecard + points table
  models.py             rule / scorecard / GBM tiers
  metrics.py            KS, PR-AUC, PSI, calibration, capture rate
  policy.py             approval-rate simulation, grade benchmark
  monitoring.py         decay, drift, retrain signal
notebooks/              00 → 06, thin: they call src, they don't contain logic
tests/test_core.py      smoke tests on synthetic data
```

## Data

LendingClub public loan data via the Kaggle mirror
[`wordsforthewise/lending-club`](https://www.kaggle.com/datasets/wordsforthewise/lending-club)
— ~2.26M funded loans (2007-06 to 2018-12) and ~27.6M declined applications.
LendingClub retired its own public download after 2018; the data is public and
historical, not audited.

---

中文说明见 [README.zh-CN.md](README.zh-CN.md)。
