# choicemodels

A general package for discrete choice model estimation and simulation, covering basic Multinomial Logit and various refinements.




## API Reference

### Classes

- `Logit()`
- `MNL()`
- `NestedLogit()`
- `MixedLogit()`


### Helper functions

- `convert_long_to_wide()`
- `convert_wide_to_long()`


### class `choicemodels.Logit()`

Based on [statsmodels.discrete.discrete_model.Logit()](http://statsmodels.sourceforge.net/0.6.0/generated/statsmodels.discrete.discrete_model.Logit.html).

Parameters include:

- `endog` (list, numpy.array, pandas.Series)  
  1-d endogenous response variable. The dependent variable.

- `exog` (list of lists, numpy.array, pandas.DataFrame)  
  Exogenous variables. This is an n-by-k array where n is the number of observations and k is the number of regressors.
  
Methods include:

- `fit(...)`  
  Returns a wrapper of estimation results, which are also saved to the model object.

- `from_formula(formula, data, ...)`
  Initializes model using a formula string.

- `predict(params, exog, ...)`  
  Returns array of fitted values.
  
  
### class `choicemodels.MNL()`

Based on [pylogit.conditional_logit.MNL()](https://github.com/timothyb0912/pylogit/blob/master/pylogit/conditional_logit.py)

Parameters include:

- `data` (pandas.DataFrame or path to CSV)

- `alt_id_col` (str)  
  Name of column containing alternative identifiers.

- `obs_id_col` (str)
  Name of column containing observation identifiers.
  
- `choice_col` (str)
  Name of column identifying whether an alternative was chosen.
  
- `specification` (OrderedDict)

Methods include:

- `fit_mle()`  
  _Returns a wrapper of estimation results,_ which are also saved to the model object.







