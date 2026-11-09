"""Classifiers Compared with Football Data (video 38), Python version.

Run from this folder:   python 38_classifiers_compared.py
The script reads ../../data/players_classification.csv and ../../data/folds_classification_2013.json (relative to this folder).
The video shows the short names "players_classification.csv" and "split.json"; only the paths differ from the code cards.
Runtime: about 30 seconds (ten models, five folds each; the grids are small). See ../README.md for the measured time.

Plan (as in the video):
  Outcome: forward (1) or not a forward (0); the positive class is 1, the class we look for.
  Training data = 2013 season, test data = 2014 season, players worth at most 10 million euros.
  Seven predictors: goals, assists, minutes, passingPerc, dribblePerc, duelAerialPerc, tacklingPerc.
  The column position gives the answer away, so it is not among the predictors.
  Ten classifiers are compared with five-fold cross-validation on the training players. Four measures per model:
  accuracy, balanced accuracy, AUC and F1. Settings are chosen by the highest AUC on the folds.
  Only the winner by AUC is evaluated once on the test season.
  The five folds are the ones R drew (set.seed(2026), vfold_cv(player_train, v = 5)), so R and Python use the same folds.

Packages (see ../README.md for versions): numpy, pandas, scikit-learn, xgboost

How the settings of the R version (tidymodels) map to Python:
  cost_complexity (rpart)  -> ccp_alpha = cost_complexity * Gini impurity of the root (alphas below)
  mtry                     -> max_features;  cost -> C;  rbf_sigma -> gamma;  neighbors -> n_neighbors
  Forests, trees and the SVM AUC are computed differently in the two libraries, so small differences from the R table are expected.
"""
import json
import warnings

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, get_scorer, make_scorer
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")   # F1 is 0/0 when a model predicts no forward; scikit-learn warns and counts it as 0

# 1. Data -----------------------------------------------------------------------------------------------------------------
d_player = pd.read_csv("../../data/players_classification.csv")
d_player = d_player[d_player["marketvalue"] <= 10_000_000]
predictors = ["goals", "assists", "minutes", "passingPerc", "dribblePerc", "duelAerialPerc", "tacklingPerc"]
player_train = d_player[d_player["season"] == 2013].reset_index(drop=True)    # 1533 players, 288 forwards
player_test = d_player[d_player["season"] == 2014].reset_index(drop=True)     # 1530 players, 307 forwards
X_train, y_train = player_train[predictors], player_train["forward"]
X_test, y_test = player_test[predictors], player_test["forward"]

# 2. Folds and measures ---------------------------------------------------------------------------------------------------
split = json.load(open("../../data/folds_classification_2013.json"))
folds = [(np.array(f["fit_rows"]) - 1, np.array(f["holdout_rows"]) - 1) for f in split["folds"]]   # R counts from 1, Python from 0
measures = {"accuracy": "accuracy", "balanced_accuracy": "balanced_accuracy", "roc_auc": "roc_auc",
            "f1": make_scorer(f1_score, pos_label=1)}

# 3. Helpers --------------------------------------------------------------------------------------------------------------
# cross_validate() runs a model on the five folds; scaled() puts standardizing in front of a model.
scaled = lambda model: make_pipeline(StandardScaler(), model)

# Not on a code card: collect the four measures into one table row.
results = []

def add_cv(name, scores):
    """Model without settings: `scores` is the result of cross_validate()."""
    results.append(dict(model=name, **{m: scores["test_" + m].mean() for m in measures}, setting=""))

def add_search(name, search):
    """Model with settings: the setting with the highest mean AUC (the one GridSearchCV refits) goes into the table."""
    i = search.best_index_
    results.append(dict(model=name, **{m: search.cv_results_["mean_test_" + m][i] for m in measures},
                        setting=", ".join(f"{k.split('__')[-1]} = {v:.6g}" for k, v in search.best_params_.items())))

# 4. The ten models ---------------------------------------------------------------------------------------------------------
# 4.1 Never a forward (base rate)
add_cv("never a forward", cross_validate(DummyClassifier(strategy="prior"), X_train, y_train, cv=folds, scoring=measures))

# 4.2 Logistic regression
lr = scaled(LogisticRegression(C=np.inf, max_iter=1000))
add_cv("logistic regression", cross_validate(lr, X_train, y_train, cv=folds, scoring=measures))

# 4.3 Classification tree
gini_root = 2 * y_train.mean() * (1 - y_train.mean())
alphas = np.logspace(-3, -1, 10) * gini_root     # the ten pruning values of the R grid, rescaled for scikit-learn
tree = DecisionTreeClassifier(min_samples_split=20, min_samples_leaf=7, random_state=1)
search = GridSearchCV(tree, {"ccp_alpha": alphas}, cv=folds, scoring=measures, refit="roc_auc")
search.fit(X_train, y_train)
add_search("tree", search)

# 4.4 Random forest
forest = RandomForestClassifier(n_estimators=500, max_features=2, random_state=1)
add_cv("random forest", cross_validate(forest, X_train, y_train, cv=folds, scoring=measures))

# 4.5 XGBoost
xgb = XGBClassifier(max_depth=3, base_score=0.5, tree_method="exact")
search = GridSearchCV(xgb, {"learning_rate": [0.01, 0.1], "n_estimators": [100, 500]},
                      cv=folds, scoring=measures, refit="roc_auc")
search.fit(X_train, y_train)
add_search("XGBoost", search)

# 4.6 K-nearest neighbors
knn = scaled(KNeighborsClassifier(weights="uniform"))
search = GridSearchCV(knn, {"kneighborsclassifier__n_neighbors": [5, 15, 25, 50, 75, 100]},
                      cv=folds, scoring=measures, refit="roc_auc")
search.fit(X_train, y_train)
add_search("KNN", search)

# 4.7 Naive Bayes and LDA
add_cv("Naive Bayes", cross_validate(scaled(GaussianNB()), X_train, y_train, cv=folds, scoring=measures))
add_cv("LDA", cross_validate(scaled(LinearDiscriminantAnalysis()), X_train, y_train, cv=folds, scoring=measures))

# 4.8 Support vector classifier (linear) and radial SVM
search = GridSearchCV(scaled(SVC(kernel="linear")), {"svc__C": [0.01, 0.1, 1, 10]},
                      cv=folds, scoring=measures, refit="roc_auc")
search.fit(X_train, y_train)
add_search("linear SVM", search)

search = GridSearchCV(scaled(SVC(kernel="rbf")), {"svc__C": [0.1, 1, 10], "svc__gamma": [0.01, 0.03, 0.1]},
                      cv=folds, scoring=measures, refit="roc_auc")
search.fit(X_train, y_train)
add_search("radial SVM", search)
svm_search = search

# 5. The comparison table and the winner by measure ------------------------------------------------------------------------
cv_table = pd.DataFrame(results)
print("\nCross-validation on the 2013 training players (mean of five folds)")
print(cv_table.rename(columns={"roc_auc": "auc"}).round({"accuracy": 4, "balanced_accuracy": 4, "auc": 4, "f1": 4}).to_string(index=False))
print("\nWinner by measure:")
for m in measures:
    print(f"  {m:18s} {cv_table.loc[cv_table[m].round(12).idxmax(), 'model']}")

# 6. One check on the next season: the winner by AUC, evaluated once on the 2014 test players -------------------------------
print("\nBest radial SVM setting from the folds:", svm_search.best_params_)
svm = scaled(SVC(kernel="rbf", C=10, gamma=0.03))    # the setting printed above
svm.fit(X_train, y_train)
print({name: round(float(get_scorer(m)(svm, X_test, y_test)), 4) for name, m in measures.items()})
cm = confusion_matrix(y_test, svm.predict(X_test), labels=[1, 0])
print(cm)    # rows: truth (forward, not a forward); columns: prediction (forward, not a forward)
print(f"Predicted forward: {cm[0, 0] + cm[1, 0]}, correct: {cm[0, 0]}, wrong: {cm[1, 0]}, missed forwards: {cm[0, 1]}")
