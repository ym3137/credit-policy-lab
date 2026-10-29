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
