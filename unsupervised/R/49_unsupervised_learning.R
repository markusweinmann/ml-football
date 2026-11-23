# Unsupervised Learning with Football Data in R (video 49).
#
# Run from this folder:   Rscript 49_unsupervised_learning.R
# The script reads ../../data/players_unsupervised.csv (relative to this folder). The video shows the short name
# "players_unsupervised.csv". Two differences from the code cards: the path, and the line round(sil, 2),
# which prints the silhouette values. The scatter plot goes to Rplots.pdf when the script runs with Rscript.
# Runtime: a few seconds.
#
# Plan (as in the video):
#   2014 season, players worth at most 10 million euros (1,530 players).
#   Six performance columns: goals, assists, passingPerc, dribblePerc, duelAerialPerc, tacklingPerc.
#   The column position is not used for clustering. It is only compared with the clusters at the end.
#   K-means for K = 1 to 10 on standardized columns (elbow and silhouette), then K = 3, PCA for the picture,
#   and hierarchical clustering (complete linkage) as a check.
#
# Packages: dplyr, cluster

library(dplyr)
library(cluster)
d_player <- read.csv("../../data/players_unsupervised.csv") %>%
  filter(marketvalue <= 10000000)
X <- d_player %>%
  select(goals, assists, passingPerc,
         dribblePerc, duelAerialPerc, tacklingPerc)
Z <- scale(X)
D <- dist(Z)
fits <- lapply(1:10, function(k) {
  set.seed(2026)
  kmeans(Z, centers = k, nstart = 20, iter.max = 100)
})
wcss <- sapply(fits, function(f) f$tot.withinss)
round(wcss / wcss[1], 2)
sil <- sapply(fits[2:10], function(f)
  mean(silhouette(f$cluster, D)[, 3]))
round(sil, 2)    # not on a card: the silhouette for K = 2 to 10, as in the video
km3 <- fits[[3]]
table(km3$cluster)
aggregate(X, list(cluster = km3$cluster), mean)
pca <- prcomp(X, scale. = TRUE)
summary(pca)$importance[2:3, 1:2]
plot(pca$x[, 1], pca$x[, 2], col = km3$cluster)
tab <- table(cluster = km3$cluster,
             position = d_player$position)
tab
colSums(tab)
table(cluster = fits[[2]]$cluster,
      position = d_player$position)
hc <- hclust(D, method = "complete")
hc3 <- cutree(hc, k = 3)
sort(table(hc3))
table(kmeans = km3$cluster, hclust = hc3)
table(hclust = hc3, position = d_player$position)
