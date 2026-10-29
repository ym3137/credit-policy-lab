# Field eligibility: what the model may see at the decision moment

The accepted file has ~151 columns. Roughly a third of them did not exist when
the approval decision was made. Leaving them in produces an AUC above 0.95 and
a model that cannot be deployed, because at scoring time none of those values
are known yet.

Fields fall into **three** categories, not two. The middle one is the subtle one.

---

## A. Post-origination — excluded, always

These are generated *after* money goes out. Using them is time travel.

| Field | Why it leaks |
|---|---|
| `out_prncp`, `out_prncp_inv` | Remaining balance — a paid-off loan reads 0 |
| `total_pymnt`, `total_pymnt_inv` | Cumulative repayment to date |
| `total_rec_prncp`, `total_rec_int`, `total_rec_late_fee` | Amounts actually recovered |
| `recoveries`, `collection_recovery_fee` | Only non-zero *after* charge-off — near-perfect label proxy |
| `last_pymnt_d`, `last_pymnt_amnt`, `next_pymnt_d` | Payment history |
| `last_credit_pull_d` | Post-origination bureau pull |
| `last_fico_range_low`, `last_fico_range_high` | **Most recent** FICO, not application FICO |
| `debt_settlement_flag`, `settlement_*` | Settlement is an outcome |
| `hardship_*` (all) | Hardship plans are post-origination events |
| `pymnt_plan` | Payment plan flag, set during servicing |
| `funded_amnt_inv` | Investor funding completion |
| `collection_recovery_fee` | Same as `recoveries` |

The paired FICO fields are the classic trap: `fico_range_low`/`fico_range_high`
are at application and are **legitimate**; `last_fico_range_*` are current and
are **not**. Nearly identical names, opposite eligibility.

## B. Platform decision fields — held out as benchmark, not used as features

| Field | Status |
|---|---|
| `grade`, `sub_grade` | LendingClub's own risk grade |
| `int_rate`, `installment` | Price set *from* that grade |

These are known at origination, so they are not leakage in the strict sense.
But they are **LendingClub's model output**, not applicant attributes. Feeding
them in means the model learns to imitate the platform's scorecard, and the
resulting performance says nothing about independent risk assessment.

They are reserved for the policy benchmark instead: *at the same approval rate,
does my policy produce a lower bad rate than the platform's own grade ordering?*
That comparison requires `grade` to stay out of the feature set.

## C. Application-time features — eligible

Applicant and bureau attributes as of submission: `loan_amnt`, `term`,
`purpose`, `emp_length`, `home_ownership`, `annual_inc`,
`verification_status`, `dti`, `delinq_2yrs`, `earliest_cr_line`,
`fico_range_low/high`, `inq_last_6mths`, `mths_since_last_delinq`,
`mths_since_last_record`, `open_acc`, `pub_rec`, `revol_bal`, `revol_util`,
`total_acc`, `addr_state`, `application_type`, and the `*_il_*` / `*_bc_*` /
`*_rev_*` bureau aggregates where populated.

Two cautions inside this group:

- **Coverage changes over time.** Many bureau aggregates (`open_acc_6m`,
  `total_cu_tl`, the `mo_sin_*` family) are near-empty before ~2015 and
  populated after. A feature whose *availability* is correlated with issue date
  will look predictive out-of-time for the wrong reason. Check fill rate by
  issue quarter before admitting any of them.
- **Free text** (`emp_title`, `title`, `desc`) is excluded from v1. `desc` is
  additionally only populated in early vintages, so it carries the same
  availability-drift problem.

---

## Automated check: IV as a leakage alarm

`src/leakage.py` computes IV for every candidate feature and flags anything
above `binning.iv_leakage_alarm` (default 0.50).

A single feature with IV > 0.5 against a 36-month default target is almost
never a discovery. It is almost always a field from category A that slipped
through the exclusion list. **Treat high IV as a bug report, not a result** —
that reflex is the point of the check.
