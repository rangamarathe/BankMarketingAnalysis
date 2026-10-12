# Model Tuning Guide

Model tuning is the process of selecting the model settings, called **hyperparameters**, that perform best on unseen data.

Hyperparameters are choices made before training, such as:

- Decision Tree depth
- Number of Random Forest trees
- Number of KNN neighbors
- Logistic Regression regularization
- XGBoost learning rate

They differ from model parameters, such as regression coefficients, which are learned automatically during training.

## 1. Define the business objective

First decide what the model should accomplish.

For this project:

> Rank clients so the bank can contact a smaller group while reaching as many likely subscribers as possible.

This objective determines which evaluation metric should guide tuning.

## 2. Select the primary metric

The dataset is imbalanced: only about 11% of clients subscribed.

Accuracy is therefore not a good primary metric because predicting "no" for every client already produces approximately 89% accuracy.

The combined notebook uses:

- **Average precision** as the primary tuning metric
- ROC AUC as a secondary ranking metric
- Precision, recall, F1, lift, and coverage for business interpretation

```python
scoring = "average_precision"
```

## 3. Create training and test sets

The data is divided into:

- **Training data:** used for training and hyperparameter tuning
- **Test data:** reserved for final evaluation

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=42
)
```

`stratify=y` keeps approximately the same subscriber percentage in both sets.

The test set must not be used to select hyperparameters.

## 4. Build a preprocessing pipeline

Preprocessing must happen separately inside every cross-validation fold. Otherwise, information from validation observations could leak into training.

```python
pipeline = Pipeline([
    ("preprocess", preprocessor),
    ("model", XGBClassifier(
        objective="binary:logistic",
        eval_metric="aucpr",
        tree_method="hist",
        random_state=42
    ))
])
```

The pipeline applies:

- Numeric scaling or imputation
- Categorical encoding
- Model training

## 5. Define the hyperparameter search space

For example, the XGBoost search in the notebook examines:

```python
xgb_grid = {
    "model__n_estimators": [300],
    "model__max_depth": [3, 5],
    "model__learning_rate": [0.03, 0.08],
    "model__subsample": [0.8],
    "model__colsample_bytree": [0.8],
    "model__scale_pos_weight": [1.0, positive_weight]
}
```

| Hyperparameter | Meaning |
|---|---|
| `n_estimators` | Number of boosted trees |
| `max_depth` | Maximum complexity of each tree |
| `learning_rate` | Contribution made by each new tree |
| `subsample` | Fraction of training rows used by each tree |
| `colsample_bytree` | Fraction of features considered by each tree |
| `scale_pos_weight` | Extra weight given to subscribers |

Smaller learning rates usually require more trees but can generalize better. Greater tree depth captures more complex relationships but increases the risk of overfitting.

This grid contains:

```text
1 x 2 x 2 x 1 x 1 x 2 = 8 combinations
```

With five-fold cross-validation, it requires:

```text
8 combinations x 5 folds = 40 model fits
```

## 6. Define cross-validation

The notebook uses stratified five-fold cross-validation:

```python
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)
```

The training data is divided into five folds. For each hyperparameter combination:

1. Train on four folds.
2. Validate on the remaining fold.
3. Repeat until every fold has served as validation data.
4. Average the five validation scores.

Conceptually:

```text
Fold 1: validate on A; train on B, C, D, E
Fold 2: validate on B; train on A, C, D, E
Fold 3: validate on C; train on A, B, D, E
Fold 4: validate on D; train on A, B, C, E
Fold 5: validate on E; train on A, B, C, D
```

This is more reliable than selecting settings from one validation split.

## 7. Run the grid search

```python
search = GridSearchCV(
    estimator=pipeline,
    param_grid=xgb_grid,
    scoring="average_precision",
    cv=cv,
    n_jobs=-1,
    refit=True,
    return_train_score=True
)

search.fit(X_train, y_train)
```

The search:

1. Tests every hyperparameter combination.
2. Calculates average precision in every fold.
3. Averages the fold scores.
4. Selects the combination with the highest mean score.
5. Refits that configuration on the entire training set.

## 8. Examine the selected settings

The combined notebook selected approximately:

```python
{
    "n_estimators": 300,
    "max_depth": 5,
    "learning_rate": 0.03,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "scale_pos_weight": 1.0
}
```

Its mean cross-validation average precision was approximately:

```text
0.469
```

Interpretation:

- Depth 5 allowed moderate nonlinear interactions.
- The low learning rate made each tree contribute gradually.
- Row and feature sampling provided regularization.
- `scale_pos_weight=1.0` means additional positive-class weighting did not improve cross-validated average precision.

## 9. Check cross-validation stability and overfitting

The notebook records:

- Mean validation score
- Standard deviation across folds
- Training score
- Search time

For XGBoost:

```text
Mean CV average precision:   0.469
CV standard deviation:      0.014
Training average precision: 0.571
```

The training score is higher than the validation score, indicating some overfitting. The difference is not enough by itself to reject the model, but it should be monitored.

The validation standard deviation shows how much the score changes across data partitions. Smaller values suggest more stable performance.

## 10. Compare different model families

The same process is performed for every model:

| Model | Examples of tuned parameters |
|---|---|
| Logistic Regression | `C`, penalty, class weight |
| KNN | Neighbors, distance metric, weighting |
| Decision Tree | Depth, leaf size, criterion |
| SVM | Kernel, `C`, gamma, class weight |
| Random Forest | Trees, depth, leaf size, feature sampling |
| XGBoost | Depth, learning rate, trees, sampling |

The combined notebook's cross-validation results were approximately:

| Model | Mean CV average precision |
|---|---:|
| XGBoost | **0.469** |
| Random Forest | 0.468 |
| SVM | 0.455 |
| Logistic Regression | 0.451 |
| KNN | 0.440 |
| Decision Tree | 0.439 |

XGBoost was selected without consulting the test-set results.

## 11. Evaluate the selected model on the test set

After tuning is complete, evaluate the refitted model once on the untouched test set:

```python
best_model = search.best_estimator_

test_scores = best_model.predict_proba(X_test)[:, 1]

test_ap = average_precision_score(y_test, test_scores)
test_auc = roc_auc_score(y_test, test_scores)
```

For XGBoost:

```text
Test average precision: 0.488
Test ROC AUC:           0.812
```

The test score estimates performance on new observations.

Random Forest happened to obtain slightly higher test average precision, but it should not replace XGBoost solely because of that test result. Doing so would use the test set for model selection.

## 12. Tune the business threshold separately

Hyperparameter tuning selects how the model produces scores. Threshold selection determines which scores become positive decisions.

The bank may call the top 20% of ranked clients rather than use the default 0.5 probability threshold:

```python
threshold = np.quantile(validation_scores, 0.80)
selected = scores >= threshold
```

For the combined notebook's XGBoost ranking, calling approximately the top 20% produced:

- 37.8% conversion
- 67.1% subscriber coverage
- 3.36x lift over random calling

In production, the threshold should be selected using validation data and business economics, not the final test set.

## Complete tuning workflow

```text
Define business objective
        |
        v
Choose primary metric
        |
        v
Reserve untouched test set
        |
        v
Build preprocessing/model pipeline
        |
        v
Define hyperparameter grid
        |
        v
Run stratified cross-validation
        |
        v
Select highest mean validation score
        |
        v
Check variability and overfitting
        |
        v
Refit on all training data
        |
        v
Evaluate once on test data
        |
        v
Select an operational threshold
        |
        v
Validate in a campaign pilot
```

The most important rule is:

> Hyperparameters are selected using cross-validation on the training data. The test set is used only for the final evaluation.

