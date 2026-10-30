"""Predicting Market Values with Football Data: Trees and Ensembles (video 22), Python version.

Run from this folder:   python 22_trees_and_ensembles.py
The script reads ../../data/players.csv (drops name and team) and ../../data/folds_2013.json (relative to this folder).
Runtime: about a minute or two (30 XGBoost settings x 5 folds; see ../README.md for the measured time).

Same data, same five folds and same test set as in 16_linear_models.py:
  training data = 2013 season, test data = 2014 season, players worth at most 10 million euros.
New models: regression tree, bagging, random forests (fixed and tuned) and XGBoost.

Packages (see ../README.md for versions): numpy, pandas, scikit-learn, xgboost

How the settings of the R version (tidymodels) map to Python:
  cost_complexity (rpart)   -> ccp_alpha = cost_complexity * variance of y in the fitting rows;
                               rpart's minsplit 20 -> min_samples_split=20, minbucket 7 -> min_samples_leaf=7
  mtry                      -> max_features (bagging: all 16 columns, random forest: 5)
  min_n of the forest       -> min_samples_leaf (similar to randomForest's nodesize, not identical)
  XGBoost: tree_depth -> max_depth, min_n -> min_child_weight, loss_reduction -> gamma, sample_size -> subsample,
           mtry -> colsample_bynode = mtry / 16, learn_rate -> learning_rate
Bootstrap samples and subsampling are random and differ between the libraries, so forests and XGBoost
differ slightly from the R numbers even with the same seed.
"""
import json
import warnings

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

warnings.filterwarnings("ignore")

# 1. Data and folds -----------------------------------------------------------------------------------------------------
d_player = pd.read_csv("../../data/players.csv").drop(columns=["name", "team"])   # names and clubs are not predictors
d_player = d_player[d_player["marketvalue"] <= 10_000_000]


def prep(season):
    # One season, without the season column. Position becomes two indicator columns (Defender is the baseline).
    d = d_player[d_player["season"] == season].drop(columns="season").reset_index(drop=True)
    return pd.concat([d.drop(columns="position"),
                      pd.get_dummies(d["position"], prefix="position", drop_first=True, dtype=float)], axis=1)


dtr, dte = prep(2013), prep(2014)                       # 1533 and 1530 players
y_train, y_test = dtr["marketvalue"].to_numpy(), dte["marketvalue"].to_numpy()
names = list(dtr.drop(columns="marketvalue").columns)
X_train, X_test = dtr.drop(columns="marketvalue").to_numpy(), dte.drop(columns="marketvalue").to_numpy()
P = X_train.shape[1]                                    # 16 predictor columns
assert len(dtr) == 1533 and len(dte) == 1530 and P == 16

# The five folds drawn in R (set.seed(2026), vfold_cv(player_train, v = 5)); row positions are 1-based there.
folds = [(np.array(f["fit_rows"]) - 1, np.array(f["holdout_rows"]) - 1)
         for f in json.load(open("../../data/folds_2013.json"))["folds"]]
assert len(folds) == 5


def rmse(actual, predicted):
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


# make(y_fit) returns an unfitted model; it gets the fitting targets because ccp_alpha scales with their variance.
def fold_errors(make):
    # Fit on four folds, predict the fifth, five times over.
    return [rmse(y_train[ho], make(y_train[fi]).fit(X_train[fi], y_train[fi]).predict(X_train[ho])) for fi, ho in folds]


def cv_stats(errors):
    return float(np.mean(errors)), float(np.std(errors, ddof=1) / np.sqrt(len(errors)))   # CV error and standard error


def test_error(make):
    # Fit on all 2013 players, predict the 2014 players.
    return rmse(y_test, make(y_train).fit(X_train, y_train).predict(X_test))


rows = []


def record(name, errors, make):
    cv, se = cv_stats(errors)
    rows.append(dict(model=name, cv_rmse=cv, cv_se=se, test_rmse=test_error(make)))


# 2. Regression tree ----------------------------------------------------------------------------------------------------
# cost_complexity is the price of every extra leaf. It is tuned from 10^-4 to 10^-1 (10 values).
make_tree = lambda cp: (lambda y_fit: DecisionTreeRegressor(ccp_alpha=cp * float(np.var(y_fit)),
                                                           min_samples_split=20, min_samples_leaf=7, random_state=1))
cps = 10 ** np.linspace(-4, -1, 10)
cv_tree = {cp: fold_errors(make_tree(cp)) for cp in cps}
cp = min(cps, key=lambda c: np.mean(cv_tree[c]))
print("tree: best cost_complexity", round(float(cp), 6))
record("tree", cv_tree[cp], make_tree(cp))

# 3. Bagging and random forest ------------------------------------------------------------------------------------------
# Bagging: a forest that may use all 16 columns at every split. Random forest: 5 columns (about 16 / 3).
make_forest = lambda m, leaf: (lambda y_fit: RandomForestRegressor(500, max_features=m, min_samples_leaf=leaf,
                                                                   random_state=1, n_jobs=-1))
make_bag = make_forest(1.0, 5)         # max_features=1.0 means all columns
make_rf = make_forest(5, 5)
record("bagging", fold_errors(make_bag), make_bag)
record("random_forest_mtry5", fold_errors(make_rf), make_rf)

# 4. Tuned forest: max_features (mtry) and min_samples_leaf (min_n), 5 x 5 values ------------------------------------------
# R's grid_regular(mtry(range = c(5, 15)), min_n(range = c(2, 8)), levels = 5) gives these values.
pairs = [(m, leaf) for leaf in (2, 3, 5, 6, 8) for m in (5, 7, 10, 12, 15)]
cv_forest = {pr: fold_errors(make_forest(*pr)) for pr in pairs}
best_pair = min(pairs, key=lambda pr: np.mean(cv_forest[pr]))
print("tuned forest: best (mtry, min_n)", best_pair)
record("random_forest_tuned", cv_forest[best_pair], make_forest(*best_pair))

# 5. XGBoost: 500 trees, 30 settings ------------------------------------------------------------------------------------
# The 30 settings are the ones that R draws with grid_latin_hypercube(..., size = 30) after set.seed(1).
# They are listed here so that R and Python try exactly the same settings.
# base_score is the starting prediction: the training mean. The learning rate runs from 10^-3 to 10^-0.5.
xgb_grid = [
    dict(tree_depth=14, min_n=13, loss_reduction=2.26395847781253e-09, sample_size=0.471576362978667, mtry=11, learn_rate=0.00963162066576504),
    dict(tree_depth=8, min_n=27, loss_reduction=3.32303366422785e-08, sample_size=0.619505093786865, mtry=8, learn_rate=0.0629428054508894),
    dict(tree_depth=2, min_n=26, loss_reduction=4.87991456605838e-10, sample_size=0.747167910782155, mtry=1, learn_rate=0.0892524505222666),
    dict(tree_depth=6, min_n=17, loss_reduction=9.86076952626078e-08, sample_size=0.289486214010976, mtry=8, learn_rate=0.00101516762685033),
    dict(tree_depth=7, min_n=16, loss_reduction=1.3175812455271e-08, sample_size=0.673932292731479, mtry=5, learn_rate=0.00759268843993125),
    dict(tree_depth=4, min_n=5, loss_reduction=4.31043329682684e-06, sample_size=0.603319106372073, mtry=13, learn_rate=0.042963092424915),
    dict(tree_depth=8, min_n=29, loss_reduction=0.0514440364762268, sample_size=0.418360452132765, mtry=3, learn_rate=0.0100290124365713),
    dict(tree_depth=4, min_n=18, loss_reduction=0.384185447662235, sample_size=0.669830833566375, mtry=12, learn_rate=0.213067923775443),
    dict(tree_depth=5, min_n=8, loss_reduction=0.000115616249072606, sample_size=0.564219474096317, mtry=14, learn_rate=0.0161710774040542),
    dict(tree_depth=3, min_n=19, loss_reduction=0.00959702932468418, sample_size=0.237072266673204, mtry=6, learn_rate=0.0015902357499101),
    dict(tree_depth=1, min_n=3, loss_reduction=0.0784823767704488, sample_size=0.848166309713852, mtry=10, learn_rate=0.0293498748109077),
    dict(tree_depth=9, min_n=35, loss_reduction=5.00565616853178, sample_size=0.803099122131243, mtry=7, learn_rate=0.158912727639353),
    dict(tree_depth=5, min_n=21, loss_reduction=1.83680725263785e-10, sample_size=0.185419901118148, mtry=14, learn_rate=0.00221221430440789),
    dict(tree_depth=13, min_n=22, loss_reduction=8.86625806217693e-10, sample_size=0.393918224764057, mtry=7, learn_rate=0.236102241133176),
    dict(tree_depth=9, min_n=10, loss_reduction=1.83375059282243e-07, sample_size=0.879783021402545, mtry=15, learn_rate=0.00495095217623669),
    dict(tree_depth=6, min_n=32, loss_reduction=0.377396976388961, sample_size=0.94526880193036, mtry=6, learn_rate=0.0128400062910774),
    dict(tree_depth=11, min_n=24, loss_reduction=2.75118418063609e-05, sample_size=0.924120585394558, mtry=4, learn_rate=0.00426033341843645),
    dict(tree_depth=10, min_n=39, loss_reduction=0.000335453943096087, sample_size=0.157908275157679, mtry=2, learn_rate=0.0345713572894539),
    dict(tree_depth=13, min_n=36, loss_reduction=0.0235424888571904, sample_size=0.521283542073797, mtry=4, learn_rate=0.122938816972153),
    dict(tree_depth=14, min_n=37, loss_reduction=0.933116530492482, sample_size=0.369173170297872, mtry=16, learn_rate=0.019256008072926),
    dict(tree_depth=10, min_n=13, loss_reduction=1.66532202251168e-06, sample_size=0.251798115917481, mtry=2, learn_rate=0.005767804569327),
    dict(tree_depth=6, min_n=9, loss_reduction=5.63090139535965, sample_size=0.983362335683778, mtry=12, learn_rate=0.0548827835378805),
    dict(tree_depth=11, min_n=4, loss_reduction=1.2364818242551e-05, sample_size=0.715411366261542, mtry=13, learn_rate=0.00268236440193634),
    dict(tree_depth=14, min_n=38, loss_reduction=20.523340001965, sample_size=0.900849959698971, mtry=3, learn_rate=0.0225743318867689),
    dict(tree_depth=10, min_n=27, loss_reduction=1.53053817485006e-06, sample_size=0.511105460391846, mtry=15, learn_rate=0.108849602913551),
    dict(tree_depth=15, min_n=33, loss_reduction=0.0029968903001025, sample_size=0.331104420155752, mtry=9, learn_rate=0.00184532798728952),
    dict(tree_depth=2, min_n=23, loss_reduction=6.81011702853361e-07, sample_size=0.432890484437812, mtry=10, learn_rate=0.0728405438834751),
    dict(tree_depth=7, min_n=12, loss_reduction=0.000840384470568148, sample_size=0.761006896283943, mtry=5, learn_rate=0.00129200229285677),
    dict(tree_depth=3, min_n=30, loss_reduction=0.000188147171635502, sample_size=0.209438395006582, mtry=9, learn_rate=0.264257620476028),
    dict(tree_depth=12, min_n=6, loss_reduction=5.90825628688818e-09, sample_size=0.110173275198322, mtry=11, learn_rate=0.00320691376959376),
]

base_score = float(np.mean(y_train))


def make_xgb(s):
    return lambda y_fit: xgb.XGBRegressor(n_estimators=500, base_score=base_score,
                                          max_depth=s["tree_depth"], min_child_weight=s["min_n"],
                                          gamma=s["loss_reduction"], subsample=s["sample_size"],
                                          colsample_bynode=s["mtry"] / P, learning_rate=s["learn_rate"],
                                          random_state=1, n_jobs=1,
                                          tree_method="hist", max_bin=1024)   # 'exact' does not support colsample_bynode


cv_xgb = [fold_errors(make_xgb(s)) for s in xgb_grid]
best = int(np.argmin([np.mean(e) for e in cv_xgb]))
print("xgboost: best setting", xgb_grid[best])
record("xgboost", cv_xgb[best], make_xgb(xgb_grid[best]))

# 6. Compare all models by CV error ----------------------------------------------------------------------------------------
results = pd.DataFrame(rows).sort_values("cv_rmse").reset_index(drop=True)
print(results.round(0).astype({"cv_rmse": int, "cv_se": int, "test_rmse": int}).to_string())

# 7. Variable importance of the random forest (max_features = 5) ------------------------------------------------------------
# Share of the total reduction in squared error that the splits on a variable produce; the shares add up to 1.
rf = make_rf(y_train).fit(X_train, y_train)
importance = pd.Series(rf.feature_importances_, index=names).sort_values(ascending=False)
print(importance.round(3).to_string())
