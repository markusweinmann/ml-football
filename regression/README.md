# Regression: scripts for videos 16 and 22

| Step | R | Python |
|------|---|--------|
| Video 16: linear models | `R/16_linear_models.R` | `python/16_linear_models.py` |
| Video 22: trees and ensembles | `R/22_trees_and_ensembles.R` | `python/22_trees_and_ensembles.py` |

## How to run

Open a terminal in the folder of the script. The scripts read the data with relative paths (`../../data/`).

All scripts read `data/players.csv` and drop `name` and `team` before modelling.

```
cd regression/R
Rscript 16_linear_models.R
Rscript 22_trees_and_ensembles.R
```

```
cd regression/python
python 16_linear_models.py
python 22_trees_and_ensembles.py
```

Every script runs from top to bottom and prints a table with the cross-validation error (CV RMSE, mean of five folds),
its standard error and the test error (RMSE on the 2014 season), all in euros. The CV error decides which model wins.
The test error sits next to it. The Python scripts read the same five folds as R (`data/folds_2013.json`).

## Runtime

Measured on a Mac with an Apple processor (R 4.5.1, Python 3.12).

| Script | Runtime |
|--------|---------|
| `16_linear_models.R` | about 25 seconds |
| `16_linear_models.py` | about 3 seconds |
| `22_trees_and_ensembles.R` | about 8 minutes (the XGBoost grid, 30 settings x 5 folds x 500 trees, takes most of it) |
| `22_trees_and_ensembles.py` | about 2 minutes |

## Packages

The numbers in the videos come from these versions.

R 4.5.1 with tidymodels 1.4.1, kknn 1.4.1, glmnet 4.1.10, rpart 4.1.24, randomForest 4.7-1.2, xgboost 1.7.11.1.
The package `splines` is part of R.

```
install.packages(c("tidymodels", "kknn", "glmnet", "rpart", "randomForest", "xgboost"))
```

Python 3.12 with numpy 2.5.4, pandas 3.1.0, scikit-learn 1.8.0, xgboost 2.1.4.

```
pip install numpy pandas scikit-learn==1.8.0 xgboost==2.1.4
```

Newer versions of scikit-learn and xgboost give slightly different numbers for the forests and XGBoost.
With scikit-learn 1.9.1 and xgboost 3.4.1, for example, the CV RMSE of XGBoost is 1,769,322 instead of 1,772,014 euros.

## What to expect

R and Python use the same data and folds. The mean model, the linear models, the curves in age and the tree give the
same numbers up to rounding. Ridge differs by about 15 euros. Some larger differences are expected:

- k-nearest neighbors: ties between equally distant players are broken differently (R picks 75 neighbors, Python 100).
- Lasso: scikit-learn standardizes with n, R with n - 1, so the chosen penalty differs.
- Bagging, random forests and XGBoost: bootstrap samples and subsampling are random and differ between the libraries.
  The Python XGBoost script uses the same 30 settings that R draws with `set.seed(1)`.
- `grid_latin_hypercube()` prints a deprecation warning in recent versions of dials. The code still works.
