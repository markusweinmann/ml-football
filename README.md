# ml-football

Data and code for the YouTube series "Machine Learning Explained with Football Data" by Markus Weinmann (channel @profweinmann).
Playlist: https://www.youtube.com/playlist?list=PLSNXHvg1xOMw

This repository covers the regression block and one application of the classification block, so far these videos:

- "Predicting Market Values with Football Data: Linear Models in R/Python" (video 16)
- "Predicting Market Values with Football Data: Trees and Ensembles in R/Python" (video 22)
- "Classifiers Compared with Football Data in R" and "Classifiers Compared with Football Data in Python" (video 38)

Each video comes in an R and a Python version. The scripts are in `regression/R`, `regression/python`, `classification/R` and `classification/python`.
How to run them is explained in [regression/README.md](regression/README.md) and [classification/README.md](classification/README.md).

## Data

`data/players.csv` has 3,578 rows, one row per player and season. The seasons are 2013 and 2014.
The players play in five leagues: Bundesliga, Premier League, La Liga, Serie A and Ligue 1.

| Column | Meaning |
|--------|---------|
| name | name of the player |
| team | club of the player in that season |
| season | season (2013 or 2014) |
| marketvalue | estimated market value in euros (estimates from public sources, not official figures) |
| age | age in years |
| height | height in cm |
| minutes | minutes played in the season |
| goals, assists | goals and assists in the season |
| passingPerc, dribblePerc, duelAerialPerc, tacklingPerc | rates for passing, dribbling, aerial duels and tackling, shares between 0 and 1 |
| foulsPerGame | fouls per game |
| yellowCard, redCard | number of yellow and red cards in the season |
| wikipedia | natural logarithm of the number of Wikipedia page views of the player |
| youtube | natural logarithm of the number of YouTube views for the player |
| position | Defender, Midfielder or Forward |

The regression scripts read this file. They drop `name` and `team` before modelling, because names and clubs are not predictors.
The regression videos use the 2013 season (training) and the 2014 season (test) and keep players worth at most 10 million euros.

`data/folds_2013.json` holds the five cross-validation folds of the 2013 training players. They were drawn in R
(`set.seed(2026)`, `vfold_cv(player_train, v = 5)`). The Python scripts read them, so R and Python use the same folds.

### Data for the classification video

`data/players_classification.csv` has 3,578 rows and 18 columns. It is derived from `data/players.csv` for video 38:

- The columns `name` and `team` are removed.
- The column `forward` is added: 1 if `position` is "Forward", 0 for Defender and Midfielder. This is the outcome of the classifiers.
- `marketvalue` is rounded to whole euros. All other columns are identical to `data/players.csv`.
- The column `position` stays in the file. The scripts do not use it as a predictor, because it gives the answer away.

The other columns are the same as in the table above. The file is made for teaching: it lets viewers reproduce the video.
It contains no names or clubs. It must not be used to re-identify players, for example by matching the rows with other sources.
The section "Data protection" below, including the way to have rows removed, applies to this file as well.

`data/folds_classification_2013.json` holds the five cross-validation folds of the 2013 training players for video 38
(row positions, counted from 1, within the 1,533 training players worth at most 10 million euros).
They were drawn in R (`set.seed(2026)`, `vfold_cv(player_train, v = 5)`). The Python script reads them, so R and Python use the same folds.

### Data for the causal machine learning video

`data/players_causal.csv` has 530 rows and 19 columns. It is derived from the same public sources as `data/players.csv`
and covers the seasons 2012, 2013 and 2014, for video 43 and the machine learning course at the University of Cologne:

- Selection: the players who appear in all three seasons and are worth less than 10 million euros in each of them.
  One row is one player; the outcome is the market value in the 2014 season.
- Linking: the data have no player ID. The seasons were linked by the name of the player and his age.
- Treatment: `minutes_2013`, the minutes played in 2013. A "regular" is a player with at least 2,000 minutes
  (247 of the 530 players).
- Controls: `age`, `height`, `position`, `league_2013`, `marketvalue_2012` and ten performance and popularity values
  from 2012 (`minutes_2012` to `youtube_2012`, defined as in `players.csv`). Everything is known before the 2013 minutes.
- `marketvalue` (2014) and `marketvalue_2012` are in whole euros. Divide by 1,000,000 for millions.
- The columns `name`, `team` and `season` are removed. The rows are shuffled with a fixed seed,
  so the order does not follow the order of `players.csv`.

The file is made for teaching: it lets viewers reproduce video 43 and the course session.
It contains no names or clubs. It must not be used to re-identify players, for example by matching the rows with other sources.
The section "Data protection" below, including the way to have rows removed, applies to this file as well.

### Source and license

Compiled from publicly available sources for teaching.
No license is granted for the data beyond use for learning and research.
The code is under the MIT license (see `LICENSE`).

### Data protection

The data are public professional performance statistics of football players.
They are provided for teaching and research.
The analyses in this repository do not rank or judge individual named players.
No further personal attributes will be added to the data.

If you are a person concerned and want your rows removed or pseudonymised in the current version of the data,
write to Markus Weinmann, weinmann@wiso.uni-koeln.de. The request will be handled within a reasonable time.
Copies made outside this repository (forks, archives, downloads) cannot be withdrawn.

## How to cite

Weinmann, M. (2026). *ml-football: Data and code for "Machine Learning Explained with Football Data"*.
https://github.com/markusweinmann/ml-football
