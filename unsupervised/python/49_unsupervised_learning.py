"""Unsupervised Learning with Football Data in Python (video 49).

Run from this folder:   python 49_unsupervised_learning.py
The script reads ../../data/players_unsupervised.csv (relative to this folder). The video shows the short name
"players_unsupervised.csv". Differences from the code cards: the path, print() around the lines that the video
shows as bare expressions (a script does not show them otherwise), one line that prints the silhouette values,
and plt.show(), which opens the scatter plot in a window.
Runtime: a few seconds.

Plan (as in the video):
  2014 season, players worth at most 10 million euros (1,530 players).
  Six performance columns: goals, assists, passingPerc, dribblePerc, duelAerialPerc, tacklingPerc.
  The column position is not used for clustering. It is only compared with the clusters at the end.
  K-means for K = 1 to 10 on standardized columns (elbow and silhouette), then K = 3, PCA for the picture,
  and hierarchical clustering (complete linkage) as a check.
  The clusters are numbered by their mean goals (1 = most goals), so the numbers match the video.

Packages: pandas, scikit-learn, matplotlib
"""
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
d_player = pd.read_csv("../../data/players_unsupervised.csv")
d_player = d_player[d_player["marketvalue"] <= 10_000_000]
variables = ["goals", "assists", "passingPerc",
             "dribblePerc", "duelAerialPerc", "tacklingPerc"]
X = d_player[variables]
Z = StandardScaler().fit_transform(X)
fits = [KMeans(n_clusters=k, n_init=20, random_state=2026).fit(Z)
        for k in range(1, 11)]
wcss = [f.inertia_ for f in fits]
print([round(w / wcss[0], 2) for w in wcss])
sil = [silhouette_score(Z, f.labels_) for f in fits[1:]]
print([round(x, 2) for x in sil])    # not on a card: the silhouette for K = 2 to 10, as in the video
km3 = fits[2]
goals_rank = (X["goals"].groupby(km3.labels_).mean()
              .rank(ascending=False).astype(int))
cluster = pd.Series(km3.labels_, index=X.index).map(goals_rank)
print(cluster.value_counts().sort_index())
print(X.groupby(cluster).mean())
import matplotlib.pyplot as plt
pca = PCA().fit(Z)
print(pca.explained_variance_ratio_[:2])
print(pca.explained_variance_ratio_[:2].cumsum())
scores = pca.transform(Z)
plt.scatter(scores[:, 0], scores[:, 1], c=cluster)
plt.show()
tab = pd.crosstab(cluster, d_player["position"])
print(tab)
print(tab.sum())
print(pd.crosstab(fits[1].labels_, d_player["position"]))
hc3 = AgglomerativeClustering(n_clusters=3,
                              linkage="complete").fit_predict(Z)
print(pd.Series(hc3).value_counts(ascending=True))
print(pd.crosstab(cluster, hc3))
print(pd.crosstab(hc3, d_player["position"]))
