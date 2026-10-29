# Methodology: labels, performance window, and splits

Everything in this file is a *decision*, not a fact. Each one is stated with the
reasoning behind it so a reviewer can disagree with the decision rather than
guess at it.

## 1. What counts as "bad"

LendingClub's `loan_status` has nine values. They fall into three groups, not two:

| Group | Values | Target |
|---|---|---|
| Bad | `Charged Off`, `Default`, `Does not meet the credit policy. Status:Charged Off` | 1 |
| Good | `Fully Paid`, `Does not meet the credit policy. Status:Fully Paid` | 0 |
| **Censored** | `Current`, `In Grace Period`, `Late (16-30 days)`, `Late (31-120 days)` | **dropped** |

The censored group is the trap. A loan still `Current` has not yet had the
chance to default — it is not a good customer, it is an *unresolved* one.
Mapping it to 0 silently imports a block of future defaults into the good
class and inflates every metric downstream.

The `Late` statuses are dropped rather than labelled bad for the same reason in
reverse: a loan 45 days late may still cure. Labelling it bad would make the
model predict delinquency, not charge-off, and those are different targets with
different business consequences.

## 2. Performance window

A loan cannot be labelled until it has had time to go bad. Term structure
decides how long that is.

- Raw data ends **2018-12**.
- A **36-month** loan issued in 2015-12 matures 2018-12.
- Charge-off is recognised with a lag (typically ~5 months of delinquency
  before write-off), so a maturity date exactly at the data cutoff is already
  marginal.

**Decision:** restrict the modelling sample to **36-month loans issued
2012-01 through 2015-12**. 60-month loans are excluded entirely rather than
mixed in — a 60-month loan issued in 2015 is still open at the cutoff, and
pooling two terms with different observable windows is the same censoring
error one level up.

**Consequence to state out loud:** the model is a 36-month-term model. It says
nothing about 60-month lending, which is a materially riskier product on this
platform.

## 3. Splits — out-of-time, never random

| Split | Issue window | Used for |
|---|---|---|
| Train | 2012-01 → 2014-12 | binning, WOE fitting, model training |
| Validation | 2015-01 → 2015-06 | bin merging, feature selection, threshold tuning |
| **Test** | 2015-07 → 2015-12 | touched **once**, at the end |
| Drift | 2016-01 → 2018-12 | score/feature distribution only — no target |

A random hold-out would let the model see 2015 applicants during training and
be scored on their contemporaries. That measures interpolation, not the thing
a credit model is actually asked to do: hold up on *next* quarter's applicants.
Splitting on `issue_d` is the whole point.

The drift window carries incomplete labels by construction — that is fine,
because PSI and score-distribution monitoring need no target at all. This turns
the dataset's age from a weakness into the one thing it does uniquely well:
eleven years is long enough to watch a model decay.

## 4. Known limitations — read before quoting any number from this repo

1. **Selection on approval.** Only funded loans have outcomes. Every model here
   is conditional on having passed LendingClub's own credit policy. The
   population a real approval model scores is the *applicant* population, which
   is visibly different (see `05_` notebook). Reject inference is attempted but
   **cannot be validated** — rejected applicants have no counterfactual outcome,
   ever.
2. **Expansion-era data only.** 2012–2015 contains no recession. Nothing here
   demonstrates behaviour under stress.
3. **US marketplace lending.** Not transferable to Chinese consumer finance, or
   to bank-originated credit, without re-estimation.
4. **Economics are illustrative.** The return module uses stated interest rates
   and an assumed recovery rate. It has no funding cost, no servicing cost, no
   collections model. It ranks policies against each other; it does not
   estimate profit.
5. **Platform under scrutiny.** LendingClub stopped publishing this data after
   2018 and settled an SEC action in 2018 over misrepresenting loan quality.
   Provenance is public and historical, not audited.
