# Unsupervised learning: script for video 49

| Step | R | Python |
|------|---|--------|
| Video 49: unsupervised learning | `R/49_unsupervised_learning.R` | `python/49_unsupervised_learning.py` |

YouTube titles: "Unsupervised Learning with Football Data in R" and "Unsupervised Learning with Football Data in Python".
Playlist: "Unsupervised Machine Learning, Explained with Football Data" (videos 44 to 49).

## How to run

Open a terminal in the folder of the script. The scripts read the data with a relative path (`../../data/`).

The scripts read `data/players_unsupervised.csv` (see the main README for its origin) and keep six performance columns.
The column `position` is not used for clustering. It is only compared with the clusters at the end.

```
cd unsupervised/R
Rscript 49_unsupervised_learning.R
```

```
cd unsupervised/python
python 49_unsupervised_learning.py
```

Every script runs from top to bottom and contains the code of the code cards in the video.
The only changes: the path to the data, one extra line that prints the silhouette values,
and in Python `print()` around the lines that the video shows as bare expressions.

## Runtime

Measured on a Mac with an Apple processor (R 4.5.1, Python 3.12).

| Script | Runtime |
|--------|---------|
| `49_unsupervised_learning.R` | about 1 second |
| `49_unsupervised_learning.py` | about 3 seconds |

## Packages

R 4.5.1 with dplyr and cluster.

```
install.packages(c("dplyr", "cluster"))
```

Python 3.12 with pandas 3.1.0, scikit-learn 1.9.1 and matplotlib.

```
pip install pandas scikit-learn==1.9.1 matplotlib
```

## What to expect

1,530 players worth at most 10 million euros in the 2014 season. Both scripts print the same numbers as the video:

| Output | Value |
|--------|-------|
| Within-cluster sum of squares, share of K = 1, for K = 2 and K = 3 | 0.79 and 0.68 |
| Mean silhouette for K = 2 | 0.23 |
| Cluster sizes for K = 3 | 242, 571 and 717 |
| PCA: share of variance of the first two components | 0.33 and 0.18 (together 0.50) |
| Players by position | 604 defenders, 307 forwards, 619 midfielders |
| Hierarchical clustering (complete linkage), cluster sizes | 43, 686 and 801 |

The silhouette values for K = 3 to 10 agree between R and Python except for K = 9 (R 0.15, Python 0.16).

R and Python number the clusters differently. The Python script orders the k-means clusters by mean goals,
so cluster 1 has the most goals, as in R. The hierarchical clusters keep the numbers of each library.
