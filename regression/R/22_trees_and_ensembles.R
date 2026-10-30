# Predicting Market Values with Football Data: Trees and Ensembles (video 22), R version.
#
# Run from this folder, after 00_prepare_data.R:   Rscript 22_trees_and_ensembles.R
# The script reads ../../data/players.csv (relative to this folder) and drops name and team before modelling.
# Runtime: the XGBoost grid (30 settings x 5 folds, 500 trees each) is the slow part.
# Expect several minutes in total; see ../README.md for the measured time.
#
# Same data, same five folds and same test set as in 16_linear_models.R:
#   training data = 2013 season, test data = 2014 season, players worth at most 10 million euros.
# New models: regression tree, bagging, random forests (fixed and tuned) and XGBoost.
# For the comparison with the linear models, run 16_linear_models.R and read its table next to this one.
#
# Packages (see ../README.md for versions): tidymodels, rpart, randomForest, xgboost

library(tidymodels)
library(rpart)
library(randomForest)
library(xgboost)
tidymodels_prefer()

# 1. Data and folds -----------------------------------------------------------------------------------------------------
d_player <- read.csv("../../data/players.csv") %>%
  select(-name, -team) %>%                      # names and clubs are not predictors
  filter(marketvalue <= 10000000)

player_train <- d_player %>% filter(season == 2013) %>% select(-season)   # 1533 players
player_test  <- d_player %>% filter(season == 2014) %>% select(-season)   # 1530 players

# Same seed and same call as in video 16, so the folds are the same.
set.seed(2026)
player_folds <- vfold_cv(player_train, v = 5)

# One recipe for all models: position becomes two indicator columns (16 predictors in total).
rec <- recipe(marketvalue ~ ., data = player_train) %>%
  step_dummy(all_nominal_predictors())

# Helpers (as in 16_linear_models.R) --------------------------------------------------------------------------------------
test_rmse <- function(fit) augment(fit, new_data = player_test) %>% rmse(marketvalue, .pred) %>% pull(.estimate)
results <- tibble()
record <- function(name, cv, fit) {
  results <<- bind_rows(results, tibble(model = name, cv_rmse = cv$mean, cv_se = cv$std_err, test_rmse = test_rmse(fit)))
}
cv_of <- function(wf) fit_resamples(wf, resamples = player_folds) %>% collect_metrics() %>% filter(.metric == "rmse")
best_cv <- function(res) show_best(res, metric = "rmse", n = 1)

# 2. Regression tree ----------------------------------------------------------------------------------------------------
# cost_complexity is the price of every extra leaf. It is tuned from 10^-4 to 10^-1 (10 values).
tree_wf <- workflow() %>%
  add_recipe(rec) %>%
  add_model(
    decision_tree(cost_complexity = tune()) %>%
      set_engine("rpart") %>%
      set_mode("regression"))
set.seed(1)
tree_res <- tune_grid(tree_wf, resamples = player_folds,
  grid = grid_regular(
    cost_complexity(range = c(-4, -1)),
    levels = 10))
select_best(tree_res, metric = "rmse")
record("tree", best_cv(tree_res), tree_wf %>% finalize_workflow(select_best(tree_res, metric = "rmse")) %>% fit(player_train))

# 3. Bagging: a random forest that may use all 16 columns at every split -------------------------------------------------
bag_wf <- workflow() %>%
  add_recipe(rec) %>%
  add_model(
    rand_forest(mtry = .cols(), trees = 500) %>%
      set_engine("randomForest", importance = TRUE) %>%
      set_mode("regression"))
set.seed(1); bag_cv <- cv_of(bag_wf)       # the seed fixes the bootstrap samples
set.seed(1); bag_fit <- fit(bag_wf, player_train)
record("bagging", bag_cv, bag_fit)

# 4. Random forest with mtry = 5 (about 16 / 3 columns per split) --------------------------------------------------------
rf_wf <- workflow() %>%
  add_recipe(rec) %>%
  add_model(
    rand_forest(mtry = 5, trees = 500) %>%
      set_engine("randomForest", importance = TRUE) %>%
      set_mode("regression"))
set.seed(1); rf_cv <- cv_of(rf_wf)
set.seed(1); rf_fit <- fit(rf_wf, player_train)
record("random_forest_mtry5", rf_cv, rf_fit)

# 5. Tuned forest: mtry from 5 to 15 and min_n (smallest leaf size) from 2 to 8, 5 x 5 values -----------------------------
tuned_wf <- workflow() %>%
  add_recipe(rec) %>%
  add_model(
    rand_forest(mtry = tune(), trees = 500, min_n = tune()) %>%
      set_engine("randomForest") %>%
      set_mode("regression"))
tuned_grid <- grid_regular(mtry(range = c(5, 15)),
                           min_n(range = c(2, 8)),
                           levels = 5)
set.seed(1)
tuned_res <- tune_grid(tuned_wf, resamples = player_folds, grid = tuned_grid)
select_best(tuned_res, metric = "rmse")
set.seed(1)
tuned_fit <- tuned_wf %>% finalize_workflow(select_best(tuned_res, metric = "rmse")) %>% fit(player_train)
record("random_forest_tuned", best_cv(tuned_res), tuned_fit)

# 6. XGBoost: 500 trees, 30 random settings (Latin hypercube) ----------------------------------------------------------
# base_score is the starting prediction. We start at the training mean (otherwise XGBoost starts near 0.5 euros
# and small learning rates never reach the market values). The learning rate runs from 10^-3 to 10^-0.5.
xgb_wf <- workflow() %>%
  add_recipe(rec) %>%
  add_model(
    boost_tree(trees = 500, tree_depth = tune(),
      min_n = tune(), loss_reduction = tune(),
      sample_size = tune(), mtry = tune(),
      learn_rate = tune()) %>%
      set_engine("xgboost", base_score = mean(
        player_train$marketvalue)) %>%
      set_mode("regression"))
set.seed(1)
xgb_grid <- grid_latin_hypercube(tree_depth(), min_n(),
  loss_reduction(),
  sample_size = sample_prop(),
  mtry(range = c(1, 16)),
  learn_rate(range = c(-3, -0.5)), size = 30)
set.seed(1)
xgb_res <- tune_grid(xgb_wf, resamples = player_folds, grid = xgb_grid)
select_best(xgb_res, metric = "rmse")
set.seed(1)
xgb_fit <- xgb_wf %>% finalize_workflow(select_best(xgb_res, metric = "rmse")) %>% fit(player_train)
record("xgboost", best_cv(xgb_res), xgb_fit)

# 7. Compare all models by CV error ----------------------------------------------------------------------------------------
results <- results %>% arrange(cv_rmse)
print(mutate(results, across(where(is.numeric), round)))   # whole euros

# 8. Variable importance of the random forest (mtry = 5) ------------------------------------------------------------------
# IncNodePurity: how much the splits on a variable reduce the squared error, summed over all trees.
# We divide by the sum so that the shares add up to 1.
imp <- randomForest::importance(extract_fit_engine(rf_fit))[, "IncNodePurity"]
print(round(sort(imp / sum(imp), decreasing = TRUE), 3))
