# Classification: script for video 38

| Step | R | Python |
|------|---|--------|
| Video 38: classifiers compared | `R/38_classifiers_compared.R` | `python/38_classifiers_compared.py` |

YouTube titles: "Classifiers Compared with Football Data in R" and "Classifiers Compared with Football Data in Python".

## How to run

Open a terminal in the folder of the script. The scripts read the data with relative paths (`../../data/`).

The scripts read `data/players_classification.csv` (see the main README for its origin) and keep seven performance columns.
The column `position` gives the answer away and is not used.

```
cd classification/R
Rscript 38_classifiers_compared.R
```

```
cd classification/python
python 38_classifiers_compared.py
```

Every script runs from top to bottom. It compares ten classifiers on the 2013 training players with five-fold cross-validation
and prints a table with accuracy, balanced accuracy, AUC and F1 (mean of five folds) and the chosen settings.
Then it prints the winner by each measure. The winner by AUC, the radial SVM, is evaluated once on the 2014 test players:
the four measures and the confusion matrix. The Python script reads the same five folds as R (`data/folds_classification_2013.json`).

## Runtime

Measured on a Mac with an Apple processor (R 4.5.1, Python 3.12).

| Script | Runtime |
|--------|---------|
| `38_classifiers_compared.R` | about 20 seconds |
| `38_classifiers_compared.py` | about 12 seconds |

## Packages

The numbers in the video come from these versions.

R 4.5.1 with tidymodels 1.4.1, discrim 1.0.2, ranger 0.17.0, klaR 1.7.3, kernlab 0.9.33, kknn 1.4.1, MASS 7.3.65,
xgboost 1.7.11.1 and rpart 4.1.24.

```
install.packages(c("tidymodels", "discrim", "ranger", "klaR", "kernlab", "kknn", "MASS", "xgboost", "rpart"))
```

Python 3.12 with numpy 2.5.4, pandas 3.1.0, scikit-learn 1.9.1, xgboost 3.4.1.

```
pip install numpy pandas scikit-learn==1.9.1 xgboost==3.4.1
```

Other versions of the libraries, especially of scikit-learn and xgboost, can give slightly different numbers for the forests and XGBoost.

## What to expect

Both scripts choose the radial SVM (cost 10, gamma 0.03) as the winner by AUC. It is also the winner by accuracy and F1.
XGBoost wins by balanced accuracy. On the 2014 test players the radial SVM predicts 245 forwards, 189 of them correctly
and 56 wrongly, and it misses 118 forwards.

R and Python use the same data and folds. The numbers agree in the rounded values of the video for all ten models.
Differences in later decimals are possible for the forests and the SVM, because the libraries compute them differently.
In R, `rbf_sigma` is called `gamma` in Python.
