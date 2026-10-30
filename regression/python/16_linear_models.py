"""Predicting Market Values with Football Data: Linear Models (video 16), Python version.

Run from this folder:   python 16_linear_models.py
The script reads ../../data/players.csv (drops name and team) and ../../data/folds_2013.json (relative to this folder).

Plan (same as in the R version):
  - Training data = all players of the 2013 season, test data = all players of the 2014 season.
    Both keep only players worth at most 10 million euros.
  - Inside the training data there are five folds for cross-validation (CV). They were drawn in R with
    set.seed(2026) and vfold_cv(player_train, v = 5) and are stored in folds_2013.json, so that R and Python
    use exactly the same folds.
  - Settings (polynomial degree, number of neighbors, penalty) are chosen by the lowest CV error.
  - The test data stay untouched until the models are chosen. The last table shows the test error of every model
    so that you can compare. In a real project you would look at the test data once, only for the winner.

Packages (see ../README.md for versions): numpy, pandas, scikit-learn
Small differences to the R version are expected for k-nearest neighbors (ties between equally distant players)
and for lasso (scikit-learn standardizes with n, R with n - 1).
"""
import json
import warnings

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler, SplineTransformer

warnings.filterwarnings("ignore")

# 1. Data -------------------------------------------------------------------------------------------------------------
d_player = pd.read_csv("../../data/players.csv").drop(columns=["name", "team"])   # names and clubs are not predictors
d_player = d_player[d_player["marketvalue"] <= 10_000_000]


def prep(season):
    # One season, without the season column. Position becomes two indicator columns (Defender is the baseline).
    d = d_player[d_player["season"] == season].drop(columns="season").reset_index(drop=True)
    return pd.concat([d.drop(columns="position"),
                      pd.get_dummies(d["position"], prefix="position", drop_first=True, dtype=float)], axis=1)


dtr, dte = prep(2013), prep(2014)                       # 1533 and 1530 players
y_train, y_test = dtr["marketvalue"].to_numpy(), dte["marketvalue"].to_numpy()
X_train, X_test = dtr.drop(columns="marketvalue"), dte.drop(columns="marketvalue")
cols = list(X_train.columns)                            # 16 predictor columns
assert len(dtr) == 1533 and len(dte) == 1530 and len(cols) == 16

# 2. Folds ------------------------------------------------------------------------------------------------------------
# Row positions are 1-based in the file (R), so we subtract 1.
folds = [(np.array(f["fit_rows"]) - 1, np.array(f["holdout_rows"]) - 1)
         for f in json.load(open("../../data/folds_2013.json"))["folds"]]
assert len(folds) == 5


def rmse(actual, predicted):
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


# Features per model. Each function receives the fitting rows and the rows to predict and returns two matrices.
# Everything that is learned from data (the mean of age, the knots of the spline) is learned from the fitting rows only.
def feats_cols(use):
    return lambda Xf, Xo: (Xf[use].to_numpy(), Xo[use].to_numpy())


def feats_const(Xf, Xo):
    return np.zeros((len(Xf), 1)), np.zeros((len(Xo), 1))


def feats_interaction(Xf, Xo):
    def f(X):
        base = X[["goals", "position_Forward", "position_Midfielder"]].copy()
        base["g_fw"] = X["goals"] * X["position_Forward"]          # slope of goals may differ by position
        base["g_mf"] = X["goals"] * X["position_Midfielder"]
        return base.to_numpy()
    return f(Xf), f(Xo)


def feats_poly(deg):
    def f(Xf, Xo):
        mean, sd = Xf["age"].mean(), Xf["age"].std(ddof=0)         # center and scale age: numerically stable powers

        def g(X):
            z = (X["age"] - mean) / sd
            P = np.column_stack([z ** k for k in range(1, deg + 1)])
            return np.column_stack([P, X.drop(columns="age").to_numpy()])
        return g(Xf), g(Xo)
    return f


def feats_step(breaks):
    edges = [-np.inf] + list(breaks) + [np.inf]

    def f(Xf, Xo):
        def g(X):
            # intervals closed on the right, as in R; the first group is the baseline
            D = np.column_stack([((X["age"] > lo) & (X["age"] <= hi)).astype(float)
                                 for lo, hi in zip(edges[1:-1], edges[2:])])
            return np.column_stack([D, X.drop(columns="age").to_numpy()])
        return g(Xf), g(Xo)
    return f


def feats_spline(knots):
    def f(Xf, Xo):
        lo, hi = Xf["age"].min(), Xf["age"].max()
        st = SplineTransformer(degree=3, knots=np.array([[lo]] + [[k] for k in knots] + [[hi]]),
                               include_bias=False, extrapolation="continue").fit(Xf[["age"]])
        g = lambda X: np.column_stack([st.transform(X[["age"]]), X.drop(columns="age").to_numpy()])
        return g(Xf), g(Xo)
    return f


def feats_knn(Xf, Xo):
    return Xf[["age", "goals"]].to_numpy(), Xo[["age", "goals"]].to_numpy()


# Fitting and evaluating -------------------------------------------------------------------------------------------------
def fit_predict(model_fn, feats, fit_idx, X_other, scale=False):
    Xf, Xo = feats(X_train.iloc[fit_idx], X_other)
    if scale:                                            # standardize with the fitting rows only
        sc = StandardScaler().fit(Xf)
        Xf, Xo = sc.transform(Xf), sc.transform(Xo)
    y_fit = y_train[fit_idx]
    model = model_fn(len(y_fit), float(np.std(y_fit))).fit(Xf, y_fit)
    return model.predict(Xo)


def test_error(model_fn, feats, scale=False):
    # Fit on all 2013 players, predict the 2014 players.
    return rmse(y_test, fit_predict(model_fn, feats, np.arange(len(y_train)), X_test, scale))


def fold_errors(model_fn, feats, scale=False):
    # Fit on four folds, predict the fifth, five times over.
    return [rmse(y_train[ho], fit_predict(model_fn, feats, fi, X_train.iloc[ho], scale)) for fi, ho in folds]


def cv_stats(errors):
    return float(np.mean(errors)), float(np.std(errors, ddof=1) / np.sqrt(len(errors)))   # CV error and standard error


lin = lambda n, sy: LinearRegression()
rows = []


def record(name, errors, model_fn, feats, scale=False):
    cv, se = cv_stats(errors)
    rows.append(dict(model=name, cv_rmse=cv, cv_se=se, test_rmse=test_error(model_fn, feats, scale)))


# 3. Mean model and linear models (videos 02 and 04) ------------------------------------------------------------------
mean_fn = lambda n, sy: DummyRegressor(strategy="mean")
record("mean", fold_errors(mean_fn, feats_const), mean_fn, feats_const)
for name, f in [("lm_goals", feats_cols(["goals"])),
                ("lm_goals_position", feats_cols(["goals", "position_Forward", "position_Midfielder"])),
                ("lm_goals_x_position", feats_interaction),
                ("lm_all", feats_cols(cols))]:                  # all 16 columns
    record(name, fold_errors(lin, f), lin, f)

# 4. Curves in age, each with all other variables (videos 06 and 07) ----------------------------------------------------
# Polynomial: degree 1 to 10, the degree with the lowest CV error wins.
cv_poly = {deg: fold_errors(lin, feats_poly(deg)) for deg in range(1, 11)}
cv_mean = {deg: cv_stats(cv_poly[deg])[0] for deg in cv_poly}
deg = min(cv_mean, key=lambda d: cv_mean[d])
print("polynomial: best degree", deg)
record("poly_age_all", cv_poly[deg], lin, feats_poly(deg))

# Step function: age is cut at 20, 25 and 30.
record("step_age_all", fold_errors(lin, feats_step([20, 25, 30])), lin, feats_step([20, 25, 30]))

# Cubic spline with knots at 20, 25 and 30.
record("spline_age_all", fold_errors(lin, feats_spline([20, 25, 30])), lin, feats_spline([20, 25, 30]))

# 5. K-nearest neighbors on age and goals (video 09) ---------------------------------------------------------------------
# Both predictors are standardized first, so that age and goals count equally in the distance.
Ks = [1, 2, 3, 5, 7, 10, 15, 20, 30, 50, 75, 100, 150, 200]
knn = lambda k: (lambda n, sy: KNeighborsRegressor(n_neighbors=k))
cv_knn = {k: fold_errors(knn(k), feats_knn, scale=True) for k in Ks}
k = min(Ks, key=lambda k: np.mean(cv_knn[k]))
print("knn: best k", k)
record("knn_age_goals", cv_knn[k], knn(k), feats_knn, scale=True)

# 6. Ridge and lasso (videos 13 and 14) ---------------------------------------------------------------------------------
# All 16 variables, standardized. The penalty runs over 50 values from 10^3 to 10^7 (the scale used by glmnet in R).
# scikit-learn's Ridge minimizes RSS + alpha * sum(beta^2), glmnet minimizes RSS / (2n) + lambda * ...,
# and glmnet scales y by its standard deviation. This gives alpha = n * lambda / sd(y) for ridge.
# For lasso, alpha = lambda.
grid = np.logspace(3, 7, 50)
ridge = lambda g: (lambda n, sy: Ridge(alpha=n * g / sy))
lasso = lambda g: (lambda n, sy: Lasso(alpha=g, max_iter=200000, tol=1e-9))
for name, make in [("ridge_all", ridge), ("lasso_all", lasso)]:
    cv_pen = {g: fold_errors(make(g), feats_cols(cols), scale=True) for g in grid}
    best = min(grid, key=lambda g: np.mean(cv_pen[g]))
    print(name, ": best penalty", round(float(best), 1))
    record(name, cv_pen[best], make(best), feats_cols(cols), scale=True)

# 7. Compare all models by CV error; test the winner ----------------------------------------------------------------------
# The CV error decides. The test error sits next to it.
results = pd.DataFrame(rows).sort_values("cv_rmse").reset_index(drop=True)
print(results.round(0).astype({"cv_rmse": int, "cv_se": int, "test_rmse": int}).to_string())

# The winner is the model with the lowest CV error. It is refitted on all training players
# and tested on the 2014 season, once.
winner = results["model"][0]
print("CV winner:", winner, "| test RMSE:", round(float(results["test_rmse"][0])))
