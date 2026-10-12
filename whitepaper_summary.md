# White Paper Summary

## Citation

Sergio Moro, Raul M. S. Laureano, and Paulo Cortez, **"Using Data Mining for Bank Direct Marketing: An Application of the CRISP-DM Methodology."**

Source: [`CRISP-DM-BANK.pdf`](CRISP-DM-BANK.pdf)

## Purpose and business problem

The paper examines how data mining can make bank direct-marketing campaigns more efficient. A Portuguese bank wanted to sell long-term deposits while reducing the number of telephone contacts required to obtain subscriptions. The objective was therefore not simply to classify contacts, but to identify a smaller, higher-quality set of prospective clients and use contact-center resources more effectively.

The authors frame the work using CRISP-DM: Business Understanding, Data Understanding, Data Preparation, Modeling, Evaluation, and Deployment. They emphasize that CRISP-DM is iterative rather than a single linear modeling exercise.

## Data

The original database covered **17 campaigns** conducted from May 2008 through November 2010. It contained:

- 79,354 contacts
- 59 total attributes, including the output
- 6,499 successful contacts
- An initial success rate of approximately 8%

The stored information included client characteristics, bank relationships, contact history, call details, online-banking interactions, and previous campaign outcomes. The output initially contained 12 possible contact results, including successful and unsuccessful completed contacts, cancelled calls, and scheduled follow-ups.

## Three CRISP-DM iterations

### First iteration: viability and initial goal

The first goal was to predict whether a client would subscribe to the deposit. The authors initially attempted to model the 12 possible contact outcomes with all available variables. Naive Bayes completed, but Decision Tree and SVM training encountered memory or processing-time limitations. This established the need to simplify both the target and the feature set.

### Second iteration: binary target

The authors redefined the problem as binary classification by retaining only conclusive successful and unsuccessful contacts. Ambiguous or incomplete outcomes were discarded.

This produced:

- 55,817 contacts
- 53 attributes
- 6,499 successes

Naive Bayes and Decision Tree models could then be evaluated, but the authors concluded that the number of inputs and unresolved missing data still limited performance.

### Third iteration: variable and instance selection

The authors used graphical exploration in Rattle to manually remove variables judged unlikely to influence the target. For example, sex was removed because observed success rates were similar between male and female clients. They reduced the input set by roughly half.

Rows containing missing values were also discarded, resulting in:

- 45,211 contacts
- 29 input variables plus the target
- 5,289 successes
- An 11.7% success rate

The authors acknowledge that manual univariate feature selection can overlook interactions, but the later model results supported the overall reduction strategy.

## Models and validation

The study evaluated:

- Naive Bayes
- Decision Tree
- Support Vector Machine with a Gaussian kernel

The work used the open-source R packages `rminer` for modeling and `rattle` for exploratory analysis. Each experiment used a random holdout split with two-thirds of observations for training and one-third for testing. The second and third iterations used 20 runs, and reported results were averages across runs.

## Evaluation metrics

The paper emphasizes two ranking metrics:

- **ROC AUC**, which measures class discrimination across thresholds.
- **Area under the cumulative lift curve (ALIFT)**, which reflects how effectively a model concentrates responders in the highest-ranked portions of the population.

The authors argue that lift is especially relevant to direct marketing because managers must decide how many clients to contact.

## Reported results

| CRISP-DM iteration | Model | ROC AUC | ALIFT |
|---|---|---:|---:|
| First | Naive Bayes | 0.776 | 0.687 |
| Second | Naive Bayes | 0.823 | 0.790 |
| Second | Decision Tree | 0.764 | 0.591 |
| Third | Naive Bayes | 0.870 | 0.827 |
| Third | Decision Tree | 0.868 | 0.790 |
| Third | Gaussian SVM | **0.938** | **0.887** |

The results improved across the successive CRISP-DM iterations. The third-iteration Gaussian SVM was the strongest model by both ROC AUC and ALIFT.

## Important predictors

Sensitivity analysis identified final call duration as the most influential input. Month of contact ranked second. The paper observed higher modeled success in March, June, September, and December, the final months of each quarter.

These results require careful interpretation:

- Longer calls may reflect existing client interest or the time needed to complete a subscription.
- A relationship between duration and success does not establish that deliberately prolonging calls will cause subscriptions.
- Final call duration is unavailable before a call and therefore cannot directly support pre-call contact selection.

The paper itself recognizes the deployment problem: some of its strongest inputs are generated only after a campaign contact has begun. The authors propose collecting more client information so that high-quality models might eventually be built without contact-based variables.

## Conclusions of the paper

The authors conclude that:

- Iterative CRISP-DM refinement materially improved predictive performance.
- Gaussian SVM produced a high-quality classification model.
- Model sensitivity analysis could provide managers with useful information about success patterns.
- Open-source data-mining tools could produce useful real-world models at lower project cost.
- Future research should gather more client information, reduce dependence on contact-time variables, and test the models in live campaigns with marketing managers.

## Comparison with the completed notebook

| Dimension | White paper | Completed notebook |
|---|---|---|
| Final data | 45,211 rows, 29 inputs | 41,176 rows after duplicate removal; 20 original inputs plus engineered features |
| Positive rate | 11.7% | Approximately 11.3% |
| Principal models | Naive Bayes, Decision Tree, Gaussian SVM | Logistic Regression, KNN, Decision Tree, Linear SVM, Random Forest, XGBoost, plus Gaussian SVM in duration experiment |
| Validation | Repeated 2/3 training and 1/3 testing holdout | Five-fold cross-validation plus an untouched 20% test set |
| Main selection metric | ROC AUC and ALIFT | Average precision, with ROC AUC and capacity lift also reported |
| Best pre-call ROC AUC | Not reported without contact variables | Random Forest: 0.813 |
| Best duration-aware ROC AUC | Gaussian SVM: 0.938 | XGBoost: 0.952 |

The notebook's duration-aware results are similar to or higher than the paper's headline SVM performance. This supports the paper's conclusion that duration carries substantial predictive information. However, the realistic pre-call results are lower because the notebook excludes final duration from the targeting model.

The comparison is not an exact replication because the datasets, feature sets, algorithms, and validation procedures differ. In addition, the paper's ALIFT values measure the area under the full lift curve, while the notebook's 3.30x lift measures one operating point: prioritizing approximately the top 20% of clients.

## Practical interpretation

The most defensible business conclusion is that two distinct models may be useful:

1. A **pre-call ranking model**, such as the notebook's Random Forest, to prioritize outreach using information available before contact.
2. A separately trained **in-call decision-support model** using timestamped elapsed-duration data and other information observable at a particular moment during the call.

The white paper and notebook both show that data mining can improve campaign prioritization, but neither establishes that increasing call length or changing campaign month will cause additional subscriptions. Those interventions require controlled experiments.

