# Bank Marketing Classifier Comparison

## Project overview

This project compares machine-learning models for predicting whether a bank client will
subscribe to a term deposit after a telephone marketing contact. The operational objective is
to rank clients **before calls are placed**, allowing the bank to concentrate limited agent
capacity on clients who are more likely to subscribe.

The analysis distinguishes two use cases:

1. **Pre-call targeting:** uses only information available before the call. Final call
   `duration` is excluded.
2. **Duration-aware comparison:** adds final call duration to the selected XGBoost model to
   explain the difference between the realistic pre-call results and the accompanying paper's
   reported results. This is a retrospective benchmark and is not deployable for pre-call
   targeting.

The paper's Naive Bayes, Decision Tree, and Gaussian SVM algorithms are **not trained in this
notebook**. Their published AUC and ALIFT results are entered as reference values only.

## Main project files

- [`prompt_III_combined_duration_comparison.ipynb`](prompt_III_combined_duration_comparison.ipynb):
  completed and executed analysis notebook.
- [`CRISP-DM-BANK.pdf`](CRISP-DM-BANK.pdf): accompanying white paper.
- [`whitepaper_summary.md`](whitepaper_summary.md): summary of the paper.
- [`model_tuning_guide.md`](model_tuning_guide.md): model tuning parameters.
- `data/bank-additional-full.csv`: full modelling dataset.
- `data/bank-additional.csv`: smaller sample dataset.
- `data/bank-additional-names.txt`: dataset and variable documentation.

## Business understanding

Only about **11.27%** of historical contacts resulted in a subscription. Calling every client
is costly and may create unnecessary customer contact, so the useful output is a ranked list of
clients rather than a simple yes/no prediction for every record.

The intended business workflow is:

1. Score eligible clients using information known before contact.
2. Rank clients from highest to lowest estimated likelihood of subscription.
3. Select the top portion of the list based on agent capacity and campaign economics.
4. Validate the policy against the current targeting process in a controlled pilot.

## Data

The source file contains **41,188 rows and 21 columns**, including the target `y`. After removal
of 12 exact duplicates, the cleaned data contains **41,176 rows**.

The notebook performs the following preparation:

- Converts `y` from `yes`/`no` to a binary target.
- Removes 12 exact duplicate rows.
- Keeps documented `unknown` values as explicit categorical levels. For example, `default` is
  unknown for approximately 20.9% of records, so deleting all incomplete rows would discard a
  large amount of data.
- Converts the `pdays = 999` sentinel into `previously_contacted`, then drops `pdays`.
- Excludes final call `duration` from the pre-call feature set.
- One-hot encodes categorical inputs.
- Standardizes numeric inputs inside each modelling pipeline.
- Uses a stratified 80/20 train-test split, producing 32,940 training rows and 8,236 test rows.
- Keeps the test set outside hyperparameter selection.

## Descriptive and inferential statistics

### Descriptive findings

- The positive class is uncommon: only **11.27%** of contacts subscribed.
- A successful previous campaign outcome is a strong historical signal: approximately 65% of
  those contacts subscribed, compared with about 14% after a previous failure and 9% when there
  was no previous campaign outcome.
- March, September, October, and December have relatively high historical conversion rates,
  while May has a low conversion rate despite having the largest contact volume.
- Cellular contacts converted more frequently than telephone contacts in this sample.
- Students and retired clients had relatively high subscription rates, while blue-collar and
  services groups had lower rates.
- Subscription rates generally declined as the number of calls in the current campaign
  increased.
- Clients who subscribed were contacted under noticeably different economic conditions. Their
  average `euribor3m`, `nr.employed`, and `emp.var.rate` values were lower than those of clients
  who did not subscribe.

### Concise inferential check

The notebook intentionally keeps the inferential analysis at a basic course-appropriate level.
It applies a **chi-square test of independence** to each categorical feature and reports
**Cramer's V** as the effect size. No additional numeric-distribution hypothesis tests are
included.

- **Null hypothesis:** the categorical feature and subscription outcome are independent.
- **p-value:** indicates the statistical evidence against independence. A small p-value does
  not show how large or useful the association is.
- **Cramer's V:** describes association strength. Values near 0.10 are roughly small, values
  near 0.30 are moderate, and values near 0.50 are large.

| Feature | p-value | Cramer's V | Interpretation |
|---|---:|---:|---|
| `poutcome` | < 0.001 | 0.321 | Strongest association; approximately moderate |
| `month` | < 0.001 | 0.275 | Second strongest; approaching moderate |
| `job` | < 0.001 | 0.153 | Small association |
| `contact` | < 0.001 | 0.145 | Small association |
| `day_of_week` | < 0.001 | 0.025 | Statistically significant but negligible |
| `housing` | 0.057 | 0.012 | Insufficient evidence at the 5% level |
| `loan` | 0.577 | 0.005 | Insufficient evidence of association |

Because the dataset contains more than 41,000 observations, even very small differences can
produce small p-values. Statistical significance and practical importance are therefore
interpreted together. These tests show **association, not causation**; they do not establish
that changing a contact month, channel, or client characteristic will cause a subscription.

## Models evaluated

The pre-call analysis evaluates:

- Dummy majority-class baseline
- Logistic Regression
- K-Nearest Neighbors
- Decision Tree
- Linear Support Vector Machine
- Random Forest
- XGBoost

Logistic Regression, KNN, Decision Tree, SVM, Random Forest, and XGBoost are tuned using grid
search. Model selection uses five-fold stratified cross-validation and average precision. The
SVM search uses a stratified 8,000-row training subset to keep runtime practical; the other
models use the full training set.

## Evaluation strategy

### Majority-class baseline

The baseline always predicts the most frequent outcome, no subscription. It achieves **88.73%
accuracy** but **0% subscriber recall** and ROC-AUC of 0.50.

**Why it is included:** it demonstrates why high accuracy alone is misleading for an imbalanced
target. A useful model must identify subscribers, not merely reproduce the majority class.

### Average precision

Average precision summarizes the precision-recall curve across classification thresholds. It
rewards models that place actual subscribers near the top of the ranked list. Its no-skill
reference is approximately the positive rate, 0.1127.

**Why it is included:** average precision is the primary grid-search and model-selection metric
because the positive class is uncommon and the business needs a high-quality ranked list.

### ROC-AUC

ROC-AUC measures how often a model ranks a randomly selected subscriber above a randomly
selected non-subscriber. A value of 0.50 represents random ranking and 1.00 represents perfect
discrimination.

**Why it is included:** it is a standard threshold-independent ranking metric and is the main
metric reported by the paper. Because it can look optimistic when negative cases dominate, it
is secondary to average precision for model selection.

### Accuracy

Accuracy is the proportion of all predictions that are correct.

**Why it is included:** it is intuitive and part of the assignment, but it is not the primary
criterion because predicting every client as a non-subscriber is already nearly 89% accurate.

### Precision

Precision is the proportion of predicted subscribers who actually subscribe.

**Why it is included:** it represents the quality of a selected call list. Higher precision
means fewer agent calls are spent on clients who do not subscribe.

### Recall

Recall is the proportion of all actual subscribers correctly identified.

**Why it is included:** it measures subscriber coverage. Low recall means the campaign misses
many potential subscribers.

### F1 score

F1 is the harmonic mean of precision and recall.

**Why it is included:** it summarizes the threshold-dependent balance between call-list quality
and subscriber coverage, although it does not include the bank's actual financial costs.

### Cross-validation mean and standard deviation

Stratified cross-validation evaluates each parameter combination on multiple folds while
preserving approximately the same class distribution in every fold.

**Why they are included:** the mean provides a more stable basis for selecting a model than one
train-test split, while the standard deviation shows sensitivity to the particular validation
sample.

### Lift and subscriber coverage

Lift compares the conversion rate within the selected portion of the ranked list with the
overall conversion rate. Coverage is the share of all actual subscribers found within that
selected portion.

**Why they are included:** these measures translate statistical rankings into campaign-capacity
decisions. They show the benefit of calling only the highest-scored clients.

### ALIFT

ALIFT is the area under the cumulative-gains curve. Random ordering produces a value near 0.50;
higher values indicate that more subscribers appear earlier in the ranked list.

**Why it is included:** the paper reports ALIFT, so it provides a second direct reference metric
in the duration experiment. ALIFT is different from lift at one selected campaign capacity.

## Cross-validation results

| Model | Mean CV average precision | CV standard deviation |
|---|---:|---:|
| XGBoost | **0.4694** | 0.0141 |
| Random Forest | 0.4682 | 0.0167 |
| Linear SVM | 0.4552 | 0.0484 |
| Logistic Regression | 0.4514 | 0.0142 |
| K-Nearest Neighbors | 0.4395 | 0.0153 |
| Decision Tree | 0.4385 | **0.0130** |

XGBoost is selected for the targeting analysis because it has the highest mean cross-validated
average precision. Random Forest is extremely close, so the difference should not be treated as
conclusive without validation on later campaigns.

## Final pre-call holdout results

| Model | Test average precision | ROC-AUC | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|
| Random Forest | **0.4938** | **0.8166** | 0.8513 | 0.4004 | 0.6433 | **0.4936** |
| XGBoost | 0.4878 | 0.8121 | **0.8993** | 0.6402 | 0.2435 | 0.3528 |
| Decision Tree | 0.4714 | 0.7981 | 0.8039 | 0.3221 | **0.6703** | 0.4351 |
| Logistic Regression | 0.4543 | 0.8005 | 0.8985 | 0.6544 | 0.2101 | 0.3181 |
| K-Nearest Neighbors | 0.4507 | 0.7947 | 0.8984 | **0.6757** | 0.1886 | 0.2949 |
| Linear SVM | 0.4225 | 0.7795 | 0.8358 | 0.3683 | 0.6390 | 0.4673 |

Random Forest happens to have the best result on this particular holdout set. However, choosing
the final model after inspecting the test results would leak test information into model
selection. Therefore, the notebook correctly retains **XGBoost**, the cross-validation winner,
for the capacity analysis.

The precision, recall, and F1 values above use each estimator's default decision rule. That
default threshold is not automatically the best business operating point.

## Capacity-based result for the selected model

Using the cross-validation-selected XGBoost model to call the top 20% of the test list gives:

- **1,647 calls** placed.
- **623 subscribers** reached.
- **67.13%** of all subscribers in the test set reached.
- **37.83%** conversion rate among selected calls.
- **3.36x lift** over random calling.

Calling the top 10% reaches 46.88% of subscribers with a 52.86% conversion rate and 4.69x lift.
The final capacity should be selected using call cost, agent availability, expected deposit
value, consent restrictions, and the cost of missing a likely subscriber.

## Why call duration is handled separately

`duration` is the final length of a completed call. It is unknown when the bank is deciding whom
to contact, so including it in a pre-call model would leak future information.

Duration is added only for a controlled comparison because:

- The paper includes final call duration and identifies it as its most influential input.
- The experiment quantifies how much duration changes the same XGBoost model's ranking quality.
- It helps explain why a retrospective model can perform much better than a deployable pre-call
  model.

A future live in-call use case would require timestamped training examples containing only the
elapsed duration and other information available at each prediction time. It cannot be trained
using the completed call's final duration as though it were known earlier.

## Duration experiment and paper comparison

The notebook runs only XGBoost under two scenarios. Each scenario uses 20 stratified random
two-thirds/one-third train-test splits. The XGBoost settings selected earlier by pre-call
cross-validation are reused so that the primary difference is whether `duration` is present.

The paper's values are manually entered as reported references; its algorithms are not fitted.

| Result | Source | Mean AUC | Mean ALIFT |
|---|---|---:|---:|
| Paper — Naive Bayes | Published value; not fitted here | 0.870 | 0.827 |
| Paper — Decision Tree | Published value; not fitted here | 0.868 | 0.790 |
| Paper — Gaussian SVM | Published value; not fitted here | 0.938 | 0.887 |
| Our XGBoost — no duration | Mean of 20 computed runs | 0.804 | 0.770 |
| Our XGBoost — with duration | Mean of 20 computed runs | **0.950** | **0.899** |

Adding duration changes XGBoost by:

- **+0.146 AUC**
- **+0.129 ALIFT**

Compared with the paper's best reported model, Gaussian SVM:

- No-duration XGBoost is 0.134 lower in AUC and 0.117 lower in ALIFT.
- Duration-aware XGBoost is 0.012 higher in both AUC and ALIFT.

This supports the conclusion that final call duration explains a large part of the performance
difference. It does not prove that XGBoost is universally better than the paper's SVM because
the models were not evaluated on an identical dataset and feature set.

## Differences from the paper

- The paper's final modelling data contains 45,211 contacts and 29 inputs, while the supplied
  file begins with 41,188 contacts and 20 original predictors.
- The supplied data includes economic indicators that were not used in the same form in the
  paper's final dataset.
- This notebook preserves `unknown` as a categorical level, while the paper reports removing
  incomplete examples during its CRISP-DM iterations.
- This notebook selects pre-call models using stratified cross-validated average precision.
- The paper's reported algorithm results are references only. This notebook does not reproduce
  or tune the paper's Naive Bayes, Decision Tree, or Gaussian SVM models.

The comparison should therefore be interpreted as evidence about the importance of duration,
not as an exact reproduction or controlled algorithm competition.

## Main findings

- XGBoost has the highest cross-validated average precision and is selected without using the
  final test set.
- Random Forest performs slightly better on the single holdout set, showing that XGBoost and
  Random Forest are close and should be compared again using future-period data.
- Capacity-based targeting is more actionable than the default probability threshold.
- Calling the top 20% of the XGBoost-ranked test list reaches about two-thirds of subscribers
  with approximately 3.36x lift.
- Final call duration increases XGBoost AUC from 0.804 to 0.950 in the repeated comparison.
- Duration-aware performance is appropriate for retrospective comparison, not pre-call
  targeting.
- Economic conditions, previous campaign outcome, and contact timing have important predictive
  associations, but those associations are not necessarily causal.

## Recommendations

1. Pilot the pre-call XGBoost ranking against the bank's current targeting process.
2. Keep Random Forest as a close challenger during later-period validation.
3. Validate on later campaigns to test sensitivity to economic and seasonal change.
4. If an in-call model is desired, construct timestamped elapsed-call snapshots and validate it
   as a separate prediction system.



## Limitations

- A random split is not the same as testing on a later campaign period.
- The supplied dataset is not identical to the paper's final dataset.
- Paper results are transcribed reference values rather than results reproduced in this
  notebook.
- Final duration is unavailable for pre-call targeting.


