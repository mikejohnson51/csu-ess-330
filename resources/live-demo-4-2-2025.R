library(tidyverse)
library(tidymodels)

# Ingest Data
# URLs for COVID-19 case data and census population data
covid_url <-  'https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-states.csv'
pop_url   <- 'https://www2.census.gov/programs-surveys/popest/datasets/2020-2023/counties/totals/co-est2023-alldata.csv'

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
lm_mod <- linear_reg() |> 
  set_engine("lm") |> 
  set_mode("regression")

rf_model <- rand_forest() |> 
  set_engine("ranger") |> 
  set_mode("regression")

rf_model2 <- rand_forest() |> 
  set_engine("randomForest") |> 
  set_mode("regression")

b_mod <- boost_tree() |> 
  set_engine("xgboost") |> 
  set_mode("regression")

nn_mod <- mlp(hidden_units = 10) |> 
  set_engine("nnet") |> 
  set_mode("regression")

# Create Workflow Set
wf = workflow_set(list(rec), list(lm_mod, 
                                  rf_model, 
                                  rf_model2,
                                  b_mod, 
                                  nn_mod
)) |> 
  workflow_map(resamples = folds)  # Apply workflows across resamples

# Visualize Model Performance
autoplot(wf)

# Fit Selected Model (Neural Network)
fit <- workflow() |> 
  add_recipe(rec) |> 
  add_model(nn_mod) |> 
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
  #geom_label(aes(label = paste(state, season), nudge_x = 0.1, nudge_y = 0.1)) +
  labs(title = "Neural Net Model", 
       x = "Actual (Log10)", 
       y = "Predicted (Log10)") + 
  theme_minimal()

