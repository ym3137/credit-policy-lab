# Audit findings — verified against the real file

Run on the full `accepted_2007_to_2018Q4.csv.gz` (2,260,701 rows, 151 columns).
Every number below is reproducible with `notebooks/00` and `notebooks/01`.

## 1. The censoring problem is bigger than expected

| `loan_status` | Rows | Share |
|---|---:|---:|
| Fully Paid | 1,076,751 | 47.63% |
| **Current** | **878,317** | **38.85%** |
| Charged Off | 268,559 | 11.88% |
| Late (31-120 days) | 21,467 | 0.95% |
| In Grace Period | 8,436 | 0.37% |
| Late (16-30 days) | 4,349 | 0.19% |
| Does not meet credit policy: Fully Paid | 1,988 | 0.09% |
| Does not meet credit policy: Charged Off | 761 | 0.03% |
| Default | 40 | 0.00% |

**38.85% of the file is `Current`** — unresolved, not good. The common shortcut
of mapping `Current` to the good class would mislabel 878,317 loans and poison
roughly 40% of the data. This is the single highest-leverage decision in the
whole pipeline, and it is made before any model exists.

Total censored (Current + Grace + both Late states): **40.36%**.

## 2. Sample waterfall

| Step | Rows | % of raw |
|---|---:|---:|
| Raw | 2,260,701 | 100.00% |
| 36-month term only | 1,609,754 | 71.20% |
| Issued 2012-01 → 2015-12 | — | — |
| Resolved status | **589,488** | **26.08%** |

Final modelling sample: **589,488 loans, 14.03% bad rate.**

## 3. Splits — and a non-stationary target

| Split | Rows | Issue window | Bad rate |
|---|---:|---|---:|
| Train | 306,462 | 2012-01 → 2014-12 | 13.25% |
| Validation | 120,790 | 2015-01 → 2015-06 | 15.12% |
| Test | 162,236 | 2015-07 → 2015-12 | 14.71% |

Vintage curve: 13.58% (2012) → 12.33% (2013) → 13.73% (2014) → 14.89% (2015).

**The bad rate is not stable across cohorts.** Test bad rate is ~1.5 points
above train. This caps how much out-of-time performance decay can fairly be
blamed on the model: part of any observed drop is the population moving, not
the model degrading. Any decay analysis has to say which part is which.

## 4. The leakage alarm works — 9 hits, 9 correct

IV screen on train, with post-origination fields deliberately left in to test
the alarm:

| Field | IV | |
|---|---:|---|
| `last_fico_range_high` / `last_fico_range_low` | **3.1475** | post-origination |
| `last_pymnt_amnt` | 2.7244 | post-origination |
| `last_credit_pull_d` | 2.3259 | post-origination |
| `total_rec_prncp` | 1.5797 | post-origination |
| `total_pymnt_inv` / `total_pymnt` | 0.8716 / 0.8714 | post-origination |
| `debt_settlement_flag` | 0.8255 | post-origination |
| `last_pymnt_d` | 0.5549 | post-origination |

All 9 flagged fields were correctly identified as post-origination. An IV of
**3.15** against a default target is not a predictor, it is the answer key:
`last_fico_range_*` is the borrower's *most recent* score, which collapses
after a charge-off. It sits one underscore away from the legitimate
`fico_range_*` in the same file.

## 5. The finding that shapes the whole project

**No single application-time feature is strong.**

| Feature | IV |
|---|---:|
| `fico_range_low` / `fico_range_high` | 0.1247 |
| `annual_inc` | 0.0863 |
| `bc_open_to_buy` | 0.0731 |
| `avg_cur_bal` | 0.0633 |
| `tot_cur_bal` | 0.0552 |
| `acc_open_past_24mths` | 0.0536 |
| `dti` | 0.0469 |

The best legitimate single feature reaches IV 0.125 — "medium" at the bottom of
the band. Everything else is weak. Meanwhile LendingClub's own `sub_grade`
scores **0.3054**, because it is already a composite of these same inputs.

Two consequences, both of which the project should own rather than hide:

1. **Signal comes from combination, not from any one variable.** The scorecard
   has to earn its performance across many weak features. There is no hero
   predictor to find.
2. **The platform benchmark is a real opponent, not a straw man.** `sub_grade`
   at IV 0.305 is more predictive than any raw field available. Beating it at a
   fixed approval rate is a genuine test. If the scorecard only matches it,
   that is the honest result and it still demonstrates the method — a project
   that claims to crush the incumbent on this data should be distrusted.

## 6. Corrections to earlier assumptions

- `Policy Code` in the rejected file is `0.0`, **not** the 1/2 coding widely
  repeated in tutorials. Checked, not assumed.
- Both CSVs ship gzipped inside the Kaggle archive (`.csv.gz`); config paths
  point at the compressed files, which pandas reads directly.
- Chunked reads infer different dtypes per chunk (`id` arrives as int in some
  chunks, str in others). Object columns are normalised before the parquet
  write.

---

# Modelling results

Scorecard and models fitted on train (306,462), tuned on validation (120,790),
evaluated on test (162,236) **once**. Feature set: 14 features surviving the
IV floor, from 25 candidates.

## 7. The platform benchmark wins standalone — and that is the result

Test set, first and only evaluation:

| Model | AUC | PR-AUC | KS | Capture@10% |
|---|---:|---:|---:|---:|
| Tier 1 — FICO + DTI rule | 0.6208 | 0.2074 | 0.1744 | 16.7% |
| Tier 2 — WOE scorecard | 0.6660 | 0.2433 | 0.2372 | 20.0% |
| Tier 3 — gradient boosting | 0.6694 | 0.2466 | 0.2455 | 20.2% |
| **Platform `sub_grade`** | **0.6814** | **0.2515** | **0.2679** | **20.6%** |
| **Stacked: grade + scorecard** | **0.6928** | **0.2642** | **0.2795** | **21.5%** |

Approved bad rate at matched approval rates (test), versus the platform in
basis points — negative means better than the platform:

| Approval rate | Scorecard | GBM | **Stacked** |
|---:|---:|---:|---:|
| 50% | +66 bp | +57 bp | **−32 bp** |
| 60% | +61 bp | +45 bp | **−33 bp** |
| 70% | +32 bp | +23 bp | **−37 bp** |
| 80% | +19 bp | +10 bp | **−30 bp** |
| 90% | +12 bp | +9 bp | **−13 bp** |

**Built from application-time data alone, neither the scorecard nor gradient
boosting beats LendingClub's own grade.** That is the honest finding and it
should not be buried. The platform had inputs this dataset does not expose —
the full credit report, its own verification process, and visibility into the
applicants it declined.

**What does work is adding to the incumbent rather than replacing it.** The
scorecard carries information the grade does not: stacking the two beats the
grade alone by **30 to 37 basis points at every approval rate from 50% to 80%**.
Incremental lift over an existing model is also the question a risk team
actually asks about a new model, so this is the more useful answer.

The validation-to-test gap is small and the sign never flips — stacked was
−38/−35/−36/−29/−17 bp on validation and −32/−33/−37/−30/−13 bp on test. The
result held out of time, which is the only reason it is worth reporting.

### What the chosen policy costs

Stacked policy on test:

| Approval rate | Approved bad rate | Bads avoided | **Goods rejected** |
|---:|---:|---:|---:|
| 50% | 7.79% | 73.5% | **45.9%** |
| 70% | 10.08% | 52.0% | **26.2%** |
| 90% | 12.83% | 21.5% | **8.0%** |

The goods-rejected column is the one to lead with. At a 70% approval rate the
policy avoids about half the defaults and turns away **a quarter of the
customers who would have repaid**. Only 25.5% of declined applicants would
actually have defaulted. Any presentation that reports bads-avoided without
this column is selling something.

## 8. Sign-constrained pruning cost 34 basis points

Standard scorecard practice drops any WOE feature whose fitted coefficient is
non-positive, on the grounds that WOE is already oriented so higher means
riskier, and a negative coefficient means the feature is cancelling a
correlated one rather than contributing evidence.

Applied here, iterative sign-constrained pruning removed 14 of 27 features and
made the policy curve **worse by roughly 34 bp** (validation gap to platform
widened from +23 bp to +57 bp at a 50% approval rate).

So the sign constraint buys stability and auditability at a measurable cost in
performance. That trade-off is usually asserted; here it is quantified. The
shipped model uses the IV floor only, and keeps the sign diagnostic as a
reported flag rather than an automatic filter.

## 9. Calibration drifts even though the score distribution does not

Score PSI train → test is **0.0023** — the score distribution is essentially
unchanged. But calibration is systematically off:

| Decile | Predicted | Observed | Gap |
|---:|---:|---:|---:|
| 3 | 9.69% | 11.21% | +1.51 pp |
| 5 | 13.04% | 15.01% | +1.97 pp |
| 7 | 17.26% | 19.61% | +2.34 pp |
| 9 | 26.95% | 29.38% | +2.42 pp |

The model under-predicts risk across nearly every decile, because train bad
rate is 13.25% and test is 14.71%. Ranking survived; levels did not.

**This is the practical lesson of the project.** PSI monitoring alone would
have reported "stable" and missed it entirely. A policy that sets a threshold
on *probability* would have silently under-delivered on its promised bad rate,
while a policy that sets it on *approval rate* — a score quantile — is immune,
because it never relies on the absolute level being right. That is a concrete
argument for expressing credit policy as a rate rather than a probability
cut-off, and it is the kind of thing only out-of-time evaluation surfaces.

## 10. The crude economics say something uncomfortable

At a 70% approval rate on test (113,565 loans, $1.58bn deployed), the
illustrative calculation returns a net proxy of **$3.4m, or 0.22% of capital
deployed** — essentially break-even, and that is *before* funding cost,
servicing cost, and with a flat 20% recovery assumption.

This number should not be quoted as a finding; it is too crude. What it is good
for is scale discipline: marketplace consumer lending at these rates and this
loss level ran on thin margins, so a 30 bp improvement in approved bad rate is
not a rounding error in that business. It also means a resume bullet claiming
large savings from this project would not survive the first follow-up question.

---

# Stability and monitoring

Scorecard fitted on 2012–2014 train, applied to every cohort through 2018-12.
Drift window: 988,585 loans issued 2016-01 → 2018-12.

## 11. No performance decay in the observable window

| Cohort | Loans | Bad rate | AUC | KS |
|---|---:|---:|---:|---:|
| 2012 | 43,470 | 13.58% | 0.6506 | 0.2125 |
| 2013 | 100,422 | 12.33% | 0.6577 | 0.2333 |
| 2014 | 162,570 | 13.73% | 0.6570 | 0.2252 |
| 2015 | 283,026 | 14.89% | 0.6641 | 0.2361 |

Discrimination did not degrade — KS runs 0.213 → 0.236, slightly *up*. The
retrain flag never fires on this window.

**Caveat that has to travel with this table:** 2012–2014 are the training
cohorts, so those three rows are in-sample. Only 2015 is held out. The honest
statement is "no decay detected over four years", not "proven stable".

## 12. The 2018 drift is real population change, not a data artefact

Score PSI against the training baseline:

| Cohort | PSI | |
|---|---:|---|
| 2012–2016 | 0.0003 – 0.0050 | stable |
| 2017 | 0.0185 | stable |
| **2018** | **0.1006** | **watch** |

By quarter the crossing is visible: 2017Q4 0.0631 → 2018Q1 0.1040 → 2018Q3
0.1142 → 2018Q4 0.1138.

Feature-level PSI (train → 2018) locates it, and a fill-rate check settles what
it is:

| Feature | PSI | Fill: train → 2018 | Median: train → 2018 |
|---|---:|---|---|
| `percent_bc_gt_75` | 0.3159 | 0.970 → 0.986 | 50.0 → **22.2** |
| `bc_open_to_buy` | 0.2410 | 0.971 → 0.986 | 3,497 → **7,956** |
| `inq_last_6mths` | 0.1222 | 1.000 → 1.000 | mean 0.8 → **0.4** |
| `fico_range_low` | 0.1134 | 1.000 → 1.000 | 690 → **700** |

**Coverage is effectively unchanged on all four, so this is not the
availability-drift artefact the audit was built to catch.** It is a genuine
shift in who LendingClub was lending to: the 2018 book carries higher FICO
(share above 740 doubles, 7.8% → 15.9%), roughly half the bankcard utilisation,
double the available revolving credit, and half the recent credit inquiries.

The platform moved upmarket. That is a credit-policy decision by the lender,
and it is exactly the kind of change that invalidates a scorecard trained on
the older population — the model was fitted on a lower-FICO,
higher-utilisation book and is being asked to rank a cleaner one.

## 13. PSI would have missed the thing that actually broke

This is the operational conclusion of the project, and it is not the one the
textbook ordering suggests.

On the **test set**, score PSI train → test was **0.0023** — as stable as a
number can be. Calibration over the same split was off by **1.5 to 2.4
percentage points**, under-predicting risk in nine of ten deciles, because
train bad rate is 13.25% and test is 14.71%.

So a PSI-only monitoring setup reports "all clear" precisely where the model
has stopped telling the truth about absolute risk. PSI detects *input* shift;
it is blind to the target moving underneath a stable input distribution.

Two recommendations follow, and both are concrete:

1. **Monitor observed-versus-predicted bad rate by score band** on a rolling
   basis, alongside PSI rather than instead of it. Calibration drift is the
   failure mode PSI structurally cannot see.
2. **Express the approval policy as a rate, not a probability cut-off.** A
   score-quantile policy holds approved volume fixed by construction and is
   immune to calibration drift. A probability threshold silently changes
   approved volume the moment calibration moves — which here would have meant
   approving more than intended while believing the opposite.

The 2018 drift reinforces the same point from the other direction: when the
population improves, a fixed probability cut-off quietly approves a larger
share, whereas a rate-based policy keeps volume stable and surfaces the change
as a shift in realised bad rate, where it can be seen and acted on.
