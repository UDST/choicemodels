# ChoiceModels

This is a package for discrete choice model estimation and simulation, with an emphasis on large choice sets and behavioral refinements to multinomial models. Most of these models are not available in StatsModels or Scikit-learn. 


## API Reference

### Model classes

- `Logit()`
- `MNLogit()`
- `NestedLogit()`
- `MixedLogit()`


### Helper functions

- `convert_long_to_wide()`
- `convert_wide_to_long()`


### Class choicemodels.Logit()

Based on [statsmodels.discrete.discrete_model.Logit()](http://statsmodels.sourceforge.net/0.6.0/generated/statsmodels.discrete.discrete_model.Logit.html).

#### Parameters include:

- `endog` &nbsp; 1-d endogenous response variable. The dependent variable.
- `exog` &nbsp; Exogenous variables. This is an n-by-k array, for n observations and k regressors.
  
#### Methods include:

- `fit(...)` &nbsp; Returns estimation results, which are also saved to the model object.
- `from_formula(formula, data, ...)` &nbsp; Initializes model using a formula string.
- `predict(params, exog, ...)` &nbsp; Returns array of fitted values.
  
  
### Class choicemodels.MNLogit()

Based on [pylogit.conditional_logit.MNL()](https://github.com/timothyb0912/pylogit/blob/master/pylogit/conditional_logit.py).

#### Parameters include:

- `data`
- `alt_id_col` &nbsp; Name of column containing alternative identifiers.
- `obs_id_col` &nbsp; Name of column containing observation identifiers.
- `choice_col` &nbsp; Name of column identifying whether an alternative was chosen.
- `specification` &nbsp; OrderedDict

#### Methods include:

- `fit_mle(...)` &nbsp; Returns estimation results, which are also saved to the model object.
- `from_fomula(formula, data, labels, ...)` &nbsp; Not yet implemented


### Class choicemodels.CMResults()

Container for estimation results.

#### Methods include:

- `summary()` &nbsp; Returns a StatsModels-style printable summary




