library(tidyverse)
library(tidymodels)
library(bonsai)

# Ingest Data
# URLs for COVID-19 case data and census population data
covid_url <- 'https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-states.csv'
pop_url   <- 'resources/co-est2023-alldata.csv'

# Read COVID-19 case data
data = readr::read_csv(covid_url)

# Read census population data
census = readr::read_csv(pop_url) 

# Clean Census Data
census = census |> 
  filter(COUNTY == "000") |>  # Filter for state-level data only
  mutate(fips = STATE) |>      # Create a new FIPS column for merging
  select(fips, contains("2021"))  # Select relevant columns for 2021 data

# Process COVID-19 Data
state_data <-  data |> 
  group_by(fips) |> 
  mutate(
    new_cases  = pmax(0, cases - lag(cases)),   # Compute new cases, ensuring no negative values
    new_deaths = pmax(0, deaths - lag(deaths))  # Compute new deaths, ensuring no negative values
  ) |> 
  ungroup() |> 
  left_join(census, by = "fips") |>  # Merge with census data
  mutate(
    m = month(date), y = year(date),
    season = case_when(   # Define seasons based on month
      m %in% 3:5 ~ "Spring",
      m %in% 6:8 ~ "Summer",
      m %in% 9:11 ~ "Fall",
      m %in% c(12, 1, 2) ~ "Winter"
    )
  ) |> 
  group_by(state, y, season) |> 
  mutate(
    season_cases  = sum(new_cases, na.rm = TRUE),  # Aggregate seasonal cases
    season_deaths = sum(new_deaths, na.rm = TRUE)  # Aggregate seasonal deaths
  )  |> 
  distinct(state, y, season, .keep_all = TRUE) |>  # Keep only distinct rows by state, year, season
  ungroup() |> 
  select(state, contains('season'), y, POPESTIMATE2021, BIRTHS2021, DEATHS2021) |>  # Select relevant columns
  drop_na() |>  # Remove rows with missing values
  mutate(logC = log(season_cases +1))  # Log-transform case numbers for modeling

# Inspect Data Summary
skimr::skim(state_data)  # Summarize dataset

# Data Splitting for Modeling
split <- initial_split(state_data, prop = 0.8, strata = season)  # 80/20 train-test split
train <- training(split)  # Training set
test <- testing(split)  # Test set
folds <- vfold_cv(train, v = 10)  # 10-fold cross-validation

# Feature Engineering
rec = recipe(logC ~ . , data = train) |> 
  step_rm(state, season_cases) |>  # Remove non-predictive columns
  step_dummy(all_nominal()) |>  # Convert categorical variables to dummy variables
  step_scale(all_numeric_predictors()) |>  # Scale numeric predictors
  step_center(all_numeric_predictors())  # Center numeric predictors

# Define Regression Models
lm_model <- linear_reg() |> 
  set_engine("lm") |> 
  set_mode("regression")

dt_model <- decision_tree() |> 
  set_engine("rpart") |> 
  set_mode("regression")

ranger_rf_model <- rand_forest() |> 
  set_engine("ranger") |> 
  set_mode("regression")

rf_model <- rand_forest() |> 
  set_engine("randomForest") |> 
  set_mode("regression")

xgb_model <- boost_tree() |> 
  set_engine("xgboost") |> 
  set_mode("regression")

lgbm_model <- boost_tree() |> 
  set_engine("lightgbm") |> 
  set_mode("regression")

nn_model <- mlp(hidden_units = 10) |> 
  set_engine("nnet") |> 
  set_mode("regression")

# Create Workflow Set
set.seed(1)
wf <- workflow_set(list(rec), list(linear  = lm_model, 
                                  dt       = dt_model,
                                  ranger   = ranger_rf_model, 
                                  rf       = rf_model,
                                  xgb      = xgb_model, 
                                  lightgbm = lgbm_model,
                                  nnet     = nn_model)) |> 
  workflow_map(resamples = folds,
               metrics   = metric_set(mae, rsq, rmse))

# Visualize Model Performance
autoplot(wf) + 
  ggrepel::geom_label_repel(aes(label = gsub("recipe_", "", wflow_id))) + 
  theme_linedraw()

 # Select best model based on R-squared
# Fit Selected Model (Neural Network)
fit <- workflow() |> 
  add_recipe(rec) |> 
  add_model(b_mod2) |> 
  fit(data = train)

# Feature Importance
vip::vip(fit)

# Model Evaluation
predictions <- augment(fit, new_data = test) |> 
  mutate(diff = abs(logC - .pred))  # Compute absolute differences

metrics(predictions, truth = logC, estimate = .pred)  # Compute regression metrics

# Visualization of Predictions vs. Actual Values
ggplot(predictions, aes(x = logC, y = .pred)) + 
  geom_point() + 
  geom_abline() +
  labs(title = "LightGBM Model", 
       x = "Actual (Log10)", 
       y = "Predicted (Log10)", subtitle = ) + 
  theme_minimal()

###### -------------- DAY 2 ---------------------- ######

?boost_tree

b_mod_tune <- boost_tree(trees = tune(), tree_depth = tune(), min_n = tune()) |> 
  set_engine("lightgbm") |> 
  set_mode("regression")

wf_tune <-  workflow(rec, b_mod_tune) 

covid_metrics = metric_set(rsq, rmse, mae)

dials <- extract_parameter_set_dials(wf_tune) 
dials$object

my.grid <- dials |> 
  update(trees = trees(c(50, 500))) |>
  grid_latin_hypercube(size = 25)

range(my.grid$trees)

plotly::plot_ly(my.grid, 
               x = ~trees, 
               y = ~tree_depth, 
               z = ~min_n)

model_params <-  tune_grid(
  wf_tune,
  resamples = folds,
  grid = my.grid,
  metrics = covid_metrics
)

autoplot(model_params)

show_best(model_params, metric = "rsq")
show_best(model_params, metric = "rmse")
show_best(model_params, metric = "mae")

hp_best <- select_best(model_params, metric = "mae")

finalize <- finalize_workflow(wf_tune, hp_best)

final_fit <- last_fit(finalize, split, metrics = covid_metrics)

collect_metrics(final_fit)

collect_predictions(final_fit) |> 
  ggplot(aes(x = .pred, y = logC)) + 
  geom_point() +
  geom_abline() + 
  geom_smooth(method = "lm") + 
  theme_linedraw() + 
  labs(title = "Final Fit", 
       x = "Predicted (Log10)", 
       y = "Actual (Log10)")

full_pred = fit(finalize, data = state_data) |>
  augment(new_data = state_data) 

ggplot(full_pred, aes(x = .pred, y = logC, color = as.factor(y))) + 
  geom_point() +
  geom_abline() + 
  stat_ellipse() +
  geom_smooth(method = "lm") + 
  theme_linedraw() + 
  labs(title = "Final Fit", 
       x = "Predicted (Log10)", 
       y = "Actual (Log10)")

### ---- Streamlined Version ---- ####
wf_tune <- workflow(rec, 
                    boost_tree(mode       = "regression", 
                               engine     = "lightgbm", 
                               trees      = tune(), 
                               tree_depth = tune()))

set.seed(1)
hp_vars <- tune_grid(wf_tune,
                     resamples = folds,
                     grid =  25, # grid_space_filling used by default,
                     metrics = covid_metrics) |> 
  select_best(metric = "mae")

finalize <- finalize_workflow(wf_tune, hp_vars) |> 
  last_fit(split, metrics = covid_metrics)

collect_metrics(finalize)
