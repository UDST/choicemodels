# ChoiceModels change log

### 0.2.1 (2019-01-30)

- fixes a distribution error that excluded the LICENSE.txt file

### 0.2 (2019-01-25)

- production release

### 0.2.dev10 (2019-01-25)

- moves the `choicemodels.tools.distancematrix` functions directly into `choicemodels.tools`

### 0.2.dev9 (2019-01-22)

- improves documentation and packaging

### 0.2.dev8 (2019-01-21)

- prevents an infinite loop in `interative_lottery_choices()` when none of the remaining alternatives can accommodate any of the remaining choosers

### 0.2.dev7 (2018-12-12)

- adds a check to the `MergedChoiceTable` constructor to make sure there aren't any column names that overlap between the observations and alternatives tables

### 0.2.dev6 (2018-11-23)

- resolves deprecation warnings from older code

- removes `choicemodels.tools.mnl_simulate()` (originally from `urbansim.urbanchoice.mnl`), because this functionality has been fully replaced

- removes `choicemodels.Logit`, which wrapped a StatsModels estimator as proof of concept for MNL and didn't provide much value on its own

### 0.2.dev5 (2018-11-12)

- adds a `chooser_batch_size` parameter to `iterative_lottery_choices()`, to support batch simulation for very large datasets

### 0.2.dev4 (2018-10-15)

- adds a function `choicemodels.tools.iterative_lottery_choices()` for simulation of choices where the alternatives have limited capacity and choosers have varying probability distributions over the alternatives

- in `MergedChoiceTable`, empty choosers or alternatives now produces an empty choice table (rather than an exception)

- adds support for multiple tables of interaction terms in `MergedChoiceTable`

### 0.2.dev3 (2018-10-03)

- adds a function `choicemodels.tools.monte_carlo_choices()` for efficient simulation of choices for a list of scenarios that have differing probability distributions, but no capacity constraints on the alternatives

### 0.2.dev2 (2018-09-12)

- adds a `probabilities()` method to the `MultinomialLogitResults` class, which uses the fitted model coefficients to generate predicted probabilities for a table of choice scenarios

- adds a required `model_experssion` parameter to the `MultinomialLogitResults` constructor

### 0.2.dev1 (2018-08-06)

- improves the reliability of the native MNL estimator: (a) reduces the chance of a memory overflow when exponentiating utilities and (b) reports warnings from SciPy if the likelihood maximization algorithm may not have converged correctly

- adds substantial functionality to the `MergedChoiceTable` utility: sampling of alternatives with or without replacement, alternative-specific weights, interaction weights that apply to combinations of choosers and alternatives, automatic joining of interaction terms onto the merged table, non-sampling (all the alternatives available for each chooser), and estimation/simulation support for all combinations

- `LargeMultinomialLogit` class now optionally accepts a `MergedChoiceTable` as input

### 0.2.dev0 (2018-07-09)

- adds additional information to the summary table for the native MNL estimator: number of observations, df of the model, df of the residuals, rho-squared, rho-bar-squared, BIC, AIC, p values, timestamp

### 0.1.1 (2018-03-08)

- packaging improvements

### 0.1 (2018-03-08)

- initial release