# Classifiers Compared with Football Data (video 38), R version.
#
# Run from this folder:   Rscript 38_classifiers_compared.R
# The script reads ../../data/players_classification.csv (relative to this folder). The video shows the short name
# "players_classification.csv"; only the path differs from the code cards.
# Runtime: about a minute (ten models, five folds each; the grids are small). See ../README.md for the measured time.
#
# Plan (as in the video):
#   Outcome: forward (1) or not a forward (0); the positive class is 1, the class we look for.
#   Training data = 2013 season, test data = 2014 season, players worth at most 10 million euros.
#   Seven predictors: goals, assists, minutes, passingPerc, dribblePerc, duelAerialPerc, tacklingPerc.
#   The column position gives the answer away, so it is not among the columns we keep.
#   Ten classifiers are compared with five-fold cross-validation on the training players. Four measures per model:
#   accuracy, balanced accuracy, AUC and F1. Settings are chosen by the highest AUC on the folds.
#   Only the winner by AUC is evaluated once on the test season.
#
# Packages (see ../README.md for versions): tidymodels, discrim, ranger, klaR, kernlab, kknn, MASS, xgboost, rpart

library(tidymodels)
library(discrim)
tidymodels_prefer()

# 1. Data -----------------------------------------------------------------------------------------------------------------
d_player <- read.csv("../../data/players_classification.csv") %>%
  filter(marketvalue <= 10000000) %>%
  select(season, goals, assists, minutes, passingPerc, dribblePerc,
         duelAerialPerc, tacklingPerc, forward) %>%
  mutate(forward = factor(forward, levels = c("0", "1")))    # level 1 comes second: the positive class
player_train <- d_player %>% filter(season == 2013) %>% select(-season)    # 1533 players, 288 forwards
player_test  <- d_player %>% filter(season == 2014) %>% select(-season)    # 1530 players, 307 forwards
predictors <- setdiff(names(player_train), "forward")                      # the 7 predictors

# 2. Folds and measures ---------------------------------------------------------------------------------------------------
set.seed(2026)
player_folds <- vfold_cv(player_train, v = 5)
measures <- metric_set(accuracy, bal_accuracy, roc_auc, f_meas)
ctrl     <- control_grid(event_level = "second")         # for models with settings to try
ctrl_fit <- control_resamples(event_level = "second")    # for models without settings

# 3. Recipes --------------------------------------------------------------------------------------------------------------
rec     <- recipe(forward ~ ., data = player_train)
rec_std <- rec %>% step_normalize(all_predictors())      # standardizes; used for logistic regression, KNN, Bayes, LDA, SVMs

# 4. Helper that is not on a code card: collect the four measures of the best setting ---------------------------------------
# `res` is the result of fit_resamples() or tune_grid(). The setting with the highest mean AUC wins (as select_best()).
# F1 is 0/0 in a fold where a model predicts no forward at all (the never-a-forward rule). Such a fold counts as 0.
results <- list()
collect_best <- function(name, res, hyper = character(0)) {
  folds <- collect_metrics(res, summarize = FALSE) %>%
    select(id, .config, .metric, .estimate) %>%
    pivot_wider(names_from = .metric, values_from = .estimate) %>%
    mutate(f_meas = coalesce(f_meas, 0))
  by_config <- folds %>% group_by(.config) %>%
    summarise(accuracy = mean(accuracy), balanced_accuracy = mean(bal_accuracy), auc = mean(roc_auc), f1 = mean(f_meas), .groups = "drop")
  best <- by_config[which.max(round(by_config$auc, 12)), ]
  if (length(hyper) > 0) {
    chosen <- collect_metrics(res) %>% filter(.config == best$.config) %>% select(all_of(hyper)) %>% distinct()
    best$setting <- paste(names(chosen), unlist(chosen), sep = " = ", collapse = ", ")
  } else best$setting <- ""
  results[[name]] <<- best %>% mutate(model = name) %>% select(model, accuracy, balanced_accuracy, auc, f1, setting)
}
quietly <- function(x) suppressMessages(suppressWarnings(x))   # tune warns when a model predicts no forward or klaR sees a zero density

# 5. The ten models, each in a workflow of recipe plus model specification ---------------------------------------------------
# 5.1 Never a forward (base rate)
null_wf <- workflow() %>% add_recipe(rec) %>%
  add_model(null_model() %>% set_engine("parsnip") %>% set_mode("classification"))
set.seed(2026)
collect_best("never a forward", quietly(fit_resamples(null_wf, resamples = player_folds, metrics = measures, control = ctrl_fit)))

# 5.2 Logistic regression
lr_wf <- workflow() %>% add_recipe(rec_std) %>% add_model(logistic_reg() %>% set_engine("glm"))
set.seed(2026)
collect_best("logistic regression", quietly(fit_resamples(lr_wf, resamples = player_folds, metrics = measures, control = ctrl_fit)))

# 5.3 Classification tree
tree_wf <- workflow() %>% add_recipe(rec) %>%
  add_model(decision_tree(cost_complexity = tune()) %>% set_engine("rpart") %>% set_mode("classification"))
tree_grid <- grid_regular(cost_complexity(range = c(-3, -1)), levels = 10)
set.seed(2026)
tree_res <- quietly(tune_grid(tree_wf, resamples = player_folds, grid = tree_grid, metrics = measures, control = ctrl))
tree_res %>% select_best(metric = "roc_auc")             # every grid in this video: the highest AUC on the folds
collect_best("tree", tree_res, "cost_complexity")

# 5.4 Random forest
rf_wf <- workflow() %>% add_recipe(rec) %>%
  add_model(rand_forest(mtry = 2, trees = 500) %>% set_engine("ranger") %>% set_mode("classification"))
set.seed(2026)
collect_best("random forest", quietly(fit_resamples(rf_wf, resamples = player_folds, metrics = measures, control = ctrl_fit)))

# 5.5 XGBoost
xgb_wf <- workflow() %>% add_recipe(rec) %>%
  add_model(boost_tree(trees = tune(), tree_depth = 3, learn_rate = tune()) %>%
              set_engine("xgboost") %>% set_mode("classification"))
xgb_grid <- expand.grid(learn_rate = c(0.01, 0.1), trees = c(100, 500))
set.seed(2026)
collect_best("XGBoost", quietly(tune_grid(xgb_wf, resamples = player_folds, grid = xgb_grid, metrics = measures, control = ctrl)),
             c("learn_rate", "trees"))

# 5.6 K-nearest neighbors
knn_wf <- workflow() %>% add_recipe(rec_std) %>%
  add_model(nearest_neighbor(neighbors = tune(), weight_func = "rectangular") %>%
              set_engine("kknn") %>% set_mode("classification"))
knn_grid <- tibble(neighbors = c(5, 15, 25, 50, 75, 100))
set.seed(2026)
collect_best("KNN", quietly(tune_grid(knn_wf, resamples = player_folds, grid = knn_grid, metrics = measures, control = ctrl)), "neighbors")

# 5.7 Naive Bayes and LDA
nb_wf  <- workflow() %>% add_recipe(rec_std) %>% add_model(naive_Bayes() %>% set_engine("klaR", usekernel = FALSE))
lda_wf <- workflow() %>% add_recipe(rec_std) %>% add_model(discrim_linear() %>% set_engine("MASS"))
set.seed(2026)
collect_best("Naive Bayes", quietly(fit_resamples(nb_wf, resamples = player_folds, metrics = measures, control = ctrl_fit)))
set.seed(2026)
collect_best("LDA", quietly(fit_resamples(lda_wf, resamples = player_folds, metrics = measures, control = ctrl_fit)))

# 5.8 Support vector classifier (linear) and radial SVM
svml_wf <- workflow() %>% add_recipe(rec_std) %>%
  add_model(svm_linear(cost = tune()) %>% set_engine("kernlab") %>% set_mode("classification"))
svml_grid <- tibble(cost = c(0.01, 0.1, 1, 10))
set.seed(2026)
collect_best("linear SVM", quietly(tune_grid(svml_wf, resamples = player_folds, grid = svml_grid, metrics = measures, control = ctrl)), "cost")

svmr_wf <- workflow() %>% add_recipe(rec_std) %>%
  add_model(svm_rbf(cost = tune(), rbf_sigma = tune()) %>%
    set_engine("kernlab") %>% set_mode("classification"))
svmr_grid <- expand.grid(cost = c(0.1, 1, 10), rbf_sigma = c(0.01, 0.03, 0.1))
set.seed(2026)
svmr_res <- quietly(tune_grid(svmr_wf, resamples = player_folds, grid = svmr_grid,
                              metrics = measures, control = ctrl))
collect_best("radial SVM", svmr_res, c("cost", "rbf_sigma"))

# 6. The comparison table and the winner by measure ------------------------------------------------------------------------
cv_table <- bind_rows(results)
cat("\nCross-validation on the 2013 training players (mean of five folds)\n")
print(cv_table %>% mutate(across(c(accuracy, balanced_accuracy, auc, f1), ~ round(.x, 4))), n = Inf, width = Inf)
cat("\nWinner by measure:\n")
for (m in c("accuracy", "balanced_accuracy", "auc", "f1")) cat(sprintf("  %-18s %s\n", m, cv_table$model[which.max(round(cv_table[[m]], 12))]))

# 7. One check on the next season: the winner by AUC, evaluated once on the 2014 test players --------------------------------
svmr_best <- select_best(svmr_res, metric = "roc_auc")
set.seed(2026)
svm_fit <- finalize_workflow(svmr_wf, svmr_best) %>% fit(player_train)
test_pred <- augment(svm_fit, player_test)
test_pred %>% measures(forward, .pred_1, estimate = .pred_class, event_level = "second")
cm <- test_pred %>% conf_mat(forward, .pred_class)
cm
cat(sprintf("\nPredicted forward: %d, correct: %d, wrong: %d, missed forwards: %d\n",
            cm$table[2, 1] + cm$table[2, 2], cm$table[2, 2], cm$table[2, 1], cm$table[1, 2]))
