# Predicting Market Values with Football Data: Linear Models (video 16), R version.
#
# Run from this folder:   Rscript 16_linear_models.R
# The script reads ../../data/players.csv (relative to this folder) and drops name and team before modelling.
#
# Plan:
#   - Training data = all players of the 2013 season, test data = all players of the 2014 season.
#     Both keep only players worth at most 10 million euros.
#   - Inside the training data we make five folds for cross-validation (CV).
#     Every model goes through the same five folds.
#   - Settings (polynomial degree, number of neighbors, penalty) are chosen by the lowest CV error.
#   - The test data stay untouched until the models are chosen. The last table shows the test error of
#     every model so that you can compare. In a real project you would look at the test data once, only for the winner.
#
# Packages (see ../README.md for versions): tidymodels, splines, kknn, glmnet

library(tidymodels)
library(splines)
tidymodels_prefer()

# 1. Data -------------------------------------------------------------------------------------------------------------
d_player <- read.csv("../../data/players.csv") %>%
  select(-name, -team) %>%                      # names and clubs are not predictors
  filter(marketvalue <= 10000000)

player_train <- d_player %>% filter(season == 2013) %>% select(-season)   # 1533 players
player_test  <- d_player %>% filter(season == 2014) %>% select(-season)   # 1530 players

# 2. Folds ------------------------------------------------------------------------------------------------------------
# The seed fixes the random draw, so you get the same folds as in the video.
set.seed(2026)
player_folds <- vfold_cv(player_train, v = 5)

# Helpers -------------------------------------------------------------------------------------------------------------
# Every model is a workflow: a recipe (what the model sees) plus a model specification (how it learns).
lm_spec <- linear_reg() %>% set_engine("lm")
lm_wf <- function(rec) workflow() %>% add_recipe(rec) %>% add_model(lm_spec)

# Test RMSE of a fitted workflow: root mean squared error in euros.
test_rmse <- function(fit) augment(fit, new_data = player_test) %>% rmse(marketvalue, .pred) %>% pull(.estimate)

# One table row per model: CV error (mean of the five fold errors), its standard error, and the test error.
results <- tibble()
record <- function(name, cv, fit) {
  # cv is a one-row tibble with the columns mean and std_err (from collect_metrics or show_best)
  results <<- bind_rows(results, tibble(model = name, cv_rmse = cv$mean, cv_se = cv$std_err, test_rmse = test_rmse(fit)))
}
cv_of <- function(wf) fit_resamples(wf, resamples = player_folds) %>% collect_metrics() %>% filter(.metric == "rmse")
best_cv <- function(res) show_best(res, metric = "rmse", n = 1)   # CV error of the best setting

# 3. Mean model and linear models (videos 02 and 04) ------------------------------------------------------------------
# Mean model: predicts the training mean for everybody.
wf_mean <- lm_wf(recipe(marketvalue ~ 1, data = player_train))
record("mean", cv_of(wf_mean), fit(wf_mean, player_train))

# Line through goals.
wf_goals <- lm_wf(recipe(marketvalue ~ goals, data = player_train))
record("lm_goals", cv_of(wf_goals), fit(wf_goals, player_train))

# Goals plus position.
wf_gp <- lm_wf(recipe(marketvalue ~ goals + position, data = player_train))
record("lm_goals_position", cv_of(wf_gp), fit(wf_gp, player_train))

# Interaction: the slope of goals may differ by position.
wf_gxp <- lm_wf(recipe(marketvalue ~ goals + position, data = player_train) %>%
                  step_dummy(position) %>%
                  step_interact(~ goals:starts_with("position")))
record("lm_goals_x_position", cv_of(wf_gxp), fit(wf_gxp, player_train))

# All predictors (15 columns, 16 after position becomes two indicator columns).
wf_all <- lm_wf(recipe(marketvalue ~ ., data = player_train))
record("lm_all", cv_of(wf_all), fit(wf_all, player_train))

# 4. Curves in age, each with all other variables (videos 06 and 07) --------------------------------------------------
# Polynomial: the degree from 1 to 10 is tuned on the five folds.
poly_rec <- recipe(marketvalue ~ ., data = player_train) %>%
  step_poly(age, degree = tune())
poly_wf <- workflow() %>% add_model(lm_spec) %>% add_recipe(poly_rec)
poly_res <- tune_grid(poly_wf,
                      resamples = player_folds,
                      grid = grid_regular(degree(range = c(1, 10)), levels = 10))
select_best(poly_res, metric = "rmse")             # the degree with the lowest CV error
record("poly_age_all", best_cv(poly_res), poly_wf %>% finalize_workflow(select_best(poly_res, metric = "rmse")) %>% fit(player_train))

# Step function: age is cut at 20, 25 and 30.
wf_step <- workflow() %>% add_model(lm_spec) %>%
  add_recipe(recipe(marketvalue ~ ., data = player_train) %>%
               step_cut(age, breaks = c(20, 25, 30), include_outside_range = TRUE))
record("step_age_all", cv_of(wf_step), fit(wf_step, player_train))

# Cubic spline with knots at 20, 25 and 30.
wf_spline <- workflow() %>% add_model(lm_spec) %>%
  add_recipe(recipe(marketvalue ~ ., data = player_train) %>%
               step_bs(age, options = list(knots = c(20, 25, 30))))
record("spline_age_all", cv_of(wf_spline), fit(wf_spline, player_train))

# 5. K-nearest neighbors on age and goals (video 09) -----------------------------------------------------------------
# The two predictors are standardized first, so that age and goals count equally in the distance.
knn_rec <- recipe(marketvalue ~ age + goals, data = player_train) %>%
  step_normalize(all_predictors())
knn_spec <- nearest_neighbor(neighbors = tune(), weight_func = "rectangular") %>%
  set_engine("kknn") %>% set_mode("regression")
knn_wf <- workflow() %>% add_recipe(knn_rec) %>% add_model(knn_spec)
knn_grid <- tibble(neighbors = c(1, 2, 3, 5, 7, 10, 15, 20, 30, 50, 75, 100, 150, 200))
knn_res <- tune_grid(knn_wf, resamples = player_folds, grid = knn_grid)
select_best(knn_res, metric = "rmse")
record("knn_age_goals", best_cv(knn_res), knn_wf %>% finalize_workflow(select_best(knn_res, metric = "rmse")) %>% fit(player_train))

# 6. Ridge and lasso (videos 13 and 14) -------------------------------------------------------------------------------
# All variables, standardized. The penalty runs over 50 values from 10^3 to 10^7.
shrink_rec <- recipe(marketvalue ~ ., data = player_train) %>%
  step_novel(all_nominal_predictors()) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_zv(all_predictors()) %>%
  step_normalize(all_predictors())
penalty_grid <- grid_regular(penalty(range = c(3, 7)), levels = 50)

# mixture = 0 is ridge, mixture = 1 is lasso.
shrink_wf <- function(mixture) {
  workflow() %>% add_recipe(shrink_rec) %>%
    add_model(linear_reg(mixture = mixture, penalty = tune()) %>% set_engine("glmnet"))
}
ridge_res <- tune_grid(shrink_wf(0), resamples = player_folds, grid = penalty_grid)
lasso_res <- tune_grid(shrink_wf(1), resamples = player_folds, grid = penalty_grid)
select_best(ridge_res, metric = "rmse")
select_best(lasso_res, metric = "rmse")
record("ridge_all", best_cv(ridge_res), shrink_wf(0) %>% finalize_workflow(select_best(ridge_res, metric = "rmse")) %>% fit(player_train))
record("lasso_all", best_cv(lasso_res), shrink_wf(1) %>% finalize_workflow(select_best(lasso_res, metric = "rmse")) %>% fit(player_train))

# 7. Compare all models by CV error; test the winner --------------------------------------------------------------------
# The CV error decides. The test error sits next to it.
results <- results %>% arrange(cv_rmse)
print(mutate(results, across(where(is.numeric), round)), n = 20)   # whole euros

# The winner is the model with the lowest CV error. It is refitted on all training players
# and tested on the 2014 season, once.
winner <- results$model[1]
final <- poly_wf %>%
  finalize_workflow(select_best(poly_res, metric = "rmse")) %>%
  fit(player_train)
if (winner != "poly_age_all") warning("The polynomial is not the CV winner; refit the winner instead of 'final'.")
augment(final, player_test) %>%
  rmse(marketvalue, .pred)
