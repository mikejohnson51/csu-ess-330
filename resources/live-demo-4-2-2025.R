library(tidyverse)
library(tidymodels)

# Ingest
covid_url <-  'https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-states.csv'
pop_url   <- 'https://www2.census.gov/programs-surveys/popest/datasets/2020-2023/counties/totals/co-est2023-alldata.csv'

data = readr::read_csv(covid_url)

census = readr::read_csv(pop_url) 

# Clean
census = census|>
  filter(COUNTY == "000") |> 
  mutate(fips = STATE) |>
  select(fips, contains("2021"))

state_data <-  data |> 
  group_by(fips) |>
  mutate(new_cases  = pmax(0, cases - lag(cases)),
         new_deaths = pmax(0,deaths - lag(deaths))) |>
  ungroup() |>
  left_join(census, by = "fips") |>
  mutate(m = month(date), y = year(date),
         season = case_when(
           m %in% 3:5 ~ "Spring",
           m %in% 6:8 ~ "Summer",
           m %in% 9:11 ~ "Fall",
           m %in% c(12, 1, 2) ~ "Winter"
         )) |> 
  group_by(state, y, season) |>
  mutate(season_cases  = sum(new_cases, na.rm = TRUE), 
         season_deaths = sum(new_deaths, na.rm = TRUE))  |> 
  distinct(state, y, season, .keep_all = TRUE) |> 
  ungroup() |> 
  select(state, contains('season'), y, POPESTIMATE2021, BIRTHS2021, DEATHS2021) |> 
  drop_na() |> 
  mutate(logC = log(season_cases +1))

## Use this to add pmax!
skimr::skim(state_data)

# Resample
split <- initial_split(state_data, prop = 0.8, strata = season)
train <- training(split)
test <- testing(split)
folds <- vfold_cv(train, v = 10)

# Engineer 
rec = recipe(logC ~ . , data = train) |> 
  step_rm(state, season_cases) |> 
  step_dummy(all_nominal()) |>
  step_scale(all_numeric_predictors()) |> 
  step_center(all_numeric_predictors())


# Models specifications for Regression!
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


# Workflow
wf = workflow_set(list(rec), list(lm_mod, 
                                  rf_model, 
                                  rf_model2,
                                  b_mod, 
                                  nn_mod
                                )) |> 
  workflow_map(resamples = folds) 

# Select
autoplot(wf)

# Fit
fit <- workflow() |> 
  add_recipe(rec) |> 
  add_model(nn_mod) |> 
  fit(data = train)

vip::vip(fit)

# Evaluate
a <- augment(fit, new_data = test) |> 
  mutate(diff = abs(logC - .pred))

metrics(a, truth = logC, estimate = .pred)

ggplot(a, aes(x = 10^logC, y = 10^.pred)) + 
  geom_point() + 
  geom_abline() +
  geom_label(aes(label = paste(state, season), nudge_x = 0.1, nudge_y = 0.1)) +
  labs(title = "Boosted Tree Model", 
       x = "Actual", 
       y = "Predicted") + 
  theme_minimal()

