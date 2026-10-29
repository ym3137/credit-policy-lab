# Data dictionary

Two sources, both from the Kaggle mirror `wordsforthewise/lending-club`
(LendingClub's own public download was retired after 2018).

Columns are documented with **business meaning first**, because that is what a
reviewer needs in order to argue with a feature. `Available at decision` is the
eligibility flag from `docs/leakage_fields.md`.

---

## 1. `accepted_2007_to_2018Q4.csv`

~2.26M funded loans, 2007-06 → 2018-12, ~151 columns.
After the sample filter in `config.yaml` (36-month term, issued 2012–2015,
resolved status) the modelling sample is a small fraction of that — the exact
count is printed by `00_data_audit`.

### 1.1 Keys and timing

| Column | Meaning | Available at decision |
|---|---|---|
| `id` | Loan identifier | — |
| `issue_d` | Month the loan was funded, e.g. `Dec-2015`. Drives every split. | — (timing key) |
| `term` | ` 36 months` / ` 60 months`. Leading space is real; strip it. | Yes |

### 1.2 Target construction

| Column | Meaning | Available at decision |
|---|---|---|
| `loan_status` | Nine values; mapped to bad / good / censored per `docs/methodology.md` | No — this is the label |

### 1.3 Application-time features (category C)

| Column | Business meaning | Notes |
|---|---|---|
| `loan_amnt` | Amount requested | |
| `purpose` | Stated use of funds | ~14 levels; `debt_consolidation` dominates |
| `annual_inc` | Self-reported annual income | Self-reported — see `verification_status` |
| `verification_status` | Whether income was verified | Interacts with `annual_inc`; an unverified high income is weaker evidence |
| `emp_length` | Employment length, `< 1 year` … `10+ years` | Ordinal string; `n/a` is meaningful, not missing at random |
| `home_ownership` | RENT / MORTGAGE / OWN / OTHER | `ANY` and `NONE` appear rarely — collapse |
| `dti` | Debt-to-income, excluding mortgage | Sentinel values (-1, 999) exist; see audit notebook |
| `fico_range_low`, `fico_range_high` | FICO band at application | 4–5 point band; use the midpoint. **Not** `last_fico_range_*` |
| `delinq_2yrs` | 30+ day delinquencies, past 2 years | Heavily zero-inflated |
| `inq_last_6mths` | Credit inquiries, past 6 months | Proxy for credit-seeking intensity |
| `earliest_cr_line` | First credit line opened | Convert to **months of credit history as of `issue_d`**, never use the raw date |
| `open_acc` | Open credit lines | |
| `total_acc` | Total credit lines ever | |
| `pub_rec` | Derogatory public records | |
| `revol_bal` | Revolving balance | |
| `revol_util` | Revolving utilisation % | String with `%`; can exceed 100 |
| `addr_state` | State | 50 levels — bin or drop, do not one-hot naively |
| `application_type` | Individual / Joint App | Joint rows populate `annual_inc_joint`, `dti_joint` |

### 1.4 Platform decision fields (category B — benchmark only)

| Column | Meaning |
|---|---|
| `grade`, `sub_grade` | LendingClub's risk grade, A1–G5 |
| `int_rate` | Interest rate assigned from that grade |
| `installment` | Monthly payment implied by amount, term and rate |

### 1.5 Post-origination (category A — excluded)

See `docs/leakage_fields.md` for the full list and the reason for each.

---

## 2. `rejected_2007_to_2018Q4.csv`

~27.6M declined applications, 9 columns. Used only for the reject-inference
and population-shift analysis.

| Column | Business meaning | Joins to accepted? |
|---|---|---|
| `Amount Requested` | Requested amount | ≈ `loan_amnt` |
| `Application Date` | Submission date | ≈ `issue_d` (not identical — approval lag) |
| `Loan Title` | Free-text purpose | Loosely maps to `purpose`; messy |
| `Risk_Score` | FICO-like score at application | ≈ FICO midpoint, **different scale and vintage** — do not assume comparability |
| `Debt-To-Income Ratio` | DTI as a string with `%` | ≈ `dti`, different formatting |
| `Zip Code` | 3-digit prefix | ≈ `zip_code` |
| `State` | State | = `addr_state` |
| `Employment Length` | Employment length | = `emp_length` |
| `Policy Code` | **Verified against the real file: the value is `0.0`, not the 1/2 coding that is widely repeated online.** Check the distribution before filtering on it. | — |

**Only six fields overlap at all**, and `Risk_Score` is the only risk measure
among them. Any reject-inference work here is therefore a comparison on a
narrow, partially incomparable slice — which is exactly why the notebook
reports a distribution comparison and a sensitivity range rather than a
corrected model it cannot validate.

---

## 3. Filling this in

`src/data_load.py` emits a profile (dtype, fill rate, cardinality, fill rate
**by issue quarter**) for every column. Run `00_data_audit` first, then
reconcile this file against its output. A field whose fill rate jumps mid-panel
belongs in the availability-drift section of `docs/leakage_fields.md`, not in
the feature set.
