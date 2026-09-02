"""Focused PyLogit-compatible multinomial logit implementation.

The design-matrix conventions in this module are adapted from PyLogit 1.0.1,
copyright Timothy A. Brathwaite and contributors, under its BSD 3-Clause
license. See ``PYLOGIT_ATTRIBUTION.md`` and ``licenses/PYLOGIT_LICENSE.txt``.
"""

from collections import OrderedDict
from numbers import Number
import pickle
import warnings

import numpy as np
import pandas as pd
import scipy.linalg
import scipy.optimize
import scipy.stats


def create_design_matrix(data, specification, alternative_id_col, names=None):
    """Create a PyLogit-style design matrix and coefficient-name list.

    As a ChoiceModels extension, a specification entry named ``intercept``
    uses a column of ones when the input table has no such column. Name groups
    are validated against each corresponding specification entry.
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame")
    if not isinstance(specification, OrderedDict):
        raise TypeError("specification must be an OrderedDict")
    if alternative_id_col not in data:
        raise ValueError("alternative_id_col is not present in data")
    missing = [column for column in specification
               if column not in data and column != "intercept"]
    if missing:
        raise ValueError("Specification columns are missing from data: {}".format(missing))

    alternatives = np.sort(data[alternative_id_col].unique())
    columns = []
    default_names = []
    for variable, rule in specification.items():
        values = (np.ones(data.shape[0]) if variable == "intercept"
                  and variable not in data else data[variable].to_numpy())
        if rule == "all_same":
            columns.append(values)
            default_names.append(variable)
        elif rule == "all_diff":
            for alternative in alternatives:
                columns.append(values * (data[alternative_id_col].to_numpy() == alternative))
                default_names.append("{}_{}".format(variable, alternative))
        elif isinstance(rule, list):
            for group in rule:
                members = [group] if isinstance(group, Number) else list(group)
                unknown = set(members).difference(alternatives)
                if unknown:
                    raise ValueError("Unknown alternatives in specification: {}".format(unknown))
                columns.append(values * data[alternative_id_col].isin(members).to_numpy())
                default_names.append("{}_{}".format(variable, group))
        else:
            raise ValueError("Specification values must be 'all_same', 'all_diff', or lists")

    if not columns:
        raise ValueError("specification must create at least one coefficient")

    coefficient_names = default_names
    if names is not None:
        if not isinstance(names, OrderedDict) or list(names) != list(specification):
            raise ValueError("names must be an OrderedDict with the specification's keys")
        coefficient_names = []
        for variable, value in names.items():
            rule = specification[variable]
            expected = (1 if rule == "all_same" else
                        len(alternatives) if rule == "all_diff" else len(rule))
            provided = [value] if isinstance(value, str) else list(value)
            if len(provided) != expected:
                raise ValueError(
                    "names for '{}' must provide {} label(s)".format(
                        variable, expected))
            coefficient_names.extend(provided)

    return np.column_stack(columns).astype(float), coefficient_names


def _observation_slices(observation_ids):
    positions = {}
    order = []
    for position, observation_id in enumerate(observation_ids):
        if observation_id not in positions:
            positions[observation_id] = []
            order.append(observation_id)
        positions[observation_id].append(position)
    return [np.asarray(positions[observation_id], dtype=int) for observation_id in order]


def _probabilities(design, coefficients, groups):
    utilities = design.dot(coefficients)
    probabilities = np.empty(utilities.shape[0], dtype=float)
    for rows in groups:
        centered = utilities[rows] - np.max(utilities[rows])
        exponentiated = np.exp(centered)
        probabilities[rows] = exponentiated / exponentiated.sum()
    return probabilities


def predict_mnl(data, observation_id_col, alternative_id_col, specification,
                coefficients, names=None):
    """Predict long-form probabilities from a fitted flexible MNL model."""
    design, coefficient_names = create_design_matrix(
        data, specification, alternative_id_col, names)
    coefficients = np.asarray(coefficients, dtype=float)
    if coefficients.ndim != 1 or coefficients.size != design.shape[1]:
        raise ValueError("coefficients must contain one value per design column")
    groups = _observation_slices(data[observation_id_col].to_numpy())
    return _probabilities(design, coefficients, groups)


class _ModelNames(object):
    def __init__(self, endog_names, exog_names):
        self.endog_names = endog_names
        self.exog_names = exog_names


class MNLResults(object):
    """Estimation result exposing the PyLogit attributes used downstream."""

    model_type = "Multinomial Logit"

    def __init__(self, data, observation_id_col, alternative_id_col, choice_col,
                 specification, names, initial_values):
        self.data = data
        self.obs_id_col = observation_id_col
        self.alt_id_col = alternative_id_col
        self.choice_col = choice_col
        self.specification = specification
        self.name_spec = names
        self.design, self.ind_var_names = create_design_matrix(
            data, specification, alternative_id_col, names)
        self.model = _ModelNames(choice_col, list(self.ind_var_names))
        self._groups = _observation_slices(data[observation_id_col].to_numpy())
        self._choices = data[choice_col].astype(float).to_numpy()
        self._validate_data()
        if initial_values is None:
            initial_values = np.zeros(self.design.shape[1])
        elif np.isscalar(initial_values):
            initial_values = np.repeat(initial_values, self.design.shape[1])
        self._fit(np.asarray(initial_values, dtype=float))

    def _validate_data(self):
        if self.design.shape[0] != self._choices.size:
            raise ValueError("Choice and design rows do not match")
        for rows in self._groups:
            if not np.isclose(self._choices[rows].sum(), 1.0):
                raise ValueError("Each observation must have exactly one chosen alternative")

    def _fit(self, initial_values):
        if initial_values.ndim != 1 or initial_values.size != self.design.shape[1]:
            raise ValueError("initial_coefs must contain one value per coefficient")

        observation_count = float(len(self._groups))

        def objective_and_gradient(coefficients):
            probs = _probabilities(self.design, coefficients, self._groups)
            chosen = np.clip(probs[self._choices.astype(bool)], np.finfo(float).tiny, 1.0)
            objective = -np.log(chosen).sum() / observation_count
            gradient = -self.design.T.dot(self._choices - probs) / observation_count
            return objective, gradient

        fitted = scipy.optimize.minimize(
            objective_and_gradient, initial_values, jac=True, method="BFGS",
            options={"gtol": 1e-9, "maxiter": 1000})
        coefficients = fitted.x
        probabilities = _probabilities(self.design, coefficients, self._groups)

        hessian = np.zeros((coefficients.size, coefficients.size))
        score_rows = []
        for rows in self._groups:
            group_design = self.design[rows]
            group_probs = probabilities[rows]
            weights = np.diag(group_probs) - np.outer(group_probs, group_probs)
            hessian -= group_design.T.dot(weights).dot(group_design)
            score_rows.append(group_design.T.dot(self._choices[rows] - group_probs))

        covariance = scipy.linalg.pinvh(-hessian)
        scores = np.asarray(score_rows)
        robust_covariance = covariance.dot(scores.T.dot(scores)).dot(covariance)
        standard_errors = np.sqrt(np.maximum(np.diag(covariance), 0))
        robust_standard_errors = np.sqrt(np.maximum(np.diag(robust_covariance), 0))

        self.params = pd.Series(coefficients, index=self.ind_var_names, name="parameters")
        self.coefs = self.params.copy()
        self.standard_errors = pd.Series(
            standard_errors, index=self.ind_var_names, name="std_err")
        self.bse = self.standard_errors
        self.tvalues = self.params / self.standard_errors
        self.tvalues.name = "t_stats"
        self.pvalues = pd.Series(
            2 * scipy.stats.norm.sf(np.abs(self.tvalues)),
            index=self.ind_var_names, name="p_values")
        self.cov = pd.DataFrame(covariance, index=self.ind_var_names, columns=self.ind_var_names)
        self.robust_cov = pd.DataFrame(
            robust_covariance, index=self.ind_var_names, columns=self.ind_var_names)
        self.robust_std_errs = pd.Series(
            robust_standard_errors, index=self.ind_var_names, name="robust_std_err")
        self.robust_t_stats = self.params / self.robust_std_errs
        self.robust_t_stats.name = "robust_t_stats"
        self.robust_p_vals = pd.Series(
            2 * scipy.stats.norm.sf(np.abs(self.robust_t_stats)),
            index=self.ind_var_names, name="robust_p_values")
        self.summary = pd.concat([
            self.params, self.standard_errors, self.tvalues, self.pvalues,
            self.robust_std_errs, self.robust_t_stats, self.robust_p_vals], axis=1)

        self.long_fitted_probs = probabilities
        self.fitted_probs = probabilities[self._choices.astype(bool)]
        self.long_residuals = self._choices - probabilities
        mean_negative_log_likelihood = objective_and_gradient(coefficients)[0]
        self.log_likelihood = -mean_negative_log_likelihood * observation_count
        self.null_log_likelihood = -sum(np.log(len(rows)) for rows in self._groups)
        self.rho_squared = 1 - self.log_likelihood / self.null_log_likelihood
        self.rho_bar_squared = 1 - (
            self.log_likelihood - coefficients.size) / self.null_log_likelihood
        self.estimation_success = bool(fitted.success)
        self.estimation_message = str(fitted.message)
        if not self.estimation_success:
            warnings.warn(
                "MNL estimation did not converge: {}".format(self.estimation_message),
                RuntimeWarning)
        self.nobs = len(self._groups)
        self.df_model = coefficients.size
        self.df_resid = self.nobs - self.df_model
        self.llf = self.log_likelihood
        self.aic = -2 * self.log_likelihood + 2 * self.df_model
        self.bic = -2 * self.log_likelihood + np.log(self.nobs) * self.df_model

    def predict(self, data):
        design, names = create_design_matrix(
            data, self.specification, self.alt_id_col, self.name_spec)
        if names != self.ind_var_names:
            raise ValueError("Prediction design does not match the fitted model")
        return predict_mnl(
            data, self.obs_id_col, self.alt_id_col, self.specification,
            self.params.to_numpy(), self.name_spec)

    def to_pickle(self, path):
        """Write the fitted model using the method exposed by PyLogit."""
        with open(path, "wb") as stream:
            pickle.dump(self, stream, protocol=pickle.HIGHEST_PROTOCOL)

    def conf_int(self, alpha=0.05):
        critical_value = scipy.stats.norm.ppf(1 - alpha / 2.0)
        return np.column_stack([
            self.params - critical_value * self.standard_errors,
            self.params + critical_value * self.standard_errors])

    def get_statsmodels_summary(self, title=None, alpha=0.05):
        from statsmodels.iolib.summary import Summary

        result = Summary()
        result.add_table_2cols(
            self,
            title=title or "Multinomial Logit Model Regression Results",
            gleft=[("Dep. Variable:", [self.choice_col]),
                   ("Model:", [self.model_type]),
                   ("Method:", ["MLE"]),
                   ("No. Observations:", [self.nobs])],
            gright=[("Df Model:", [self.df_model]),
                    ("Df Residuals:", [self.df_resid]),
                    ("Log-Likelihood:", ["{:.3f}".format(self.log_likelihood)]),
                    ("Pseudo R-squ.:", ["{:.3f}".format(self.rho_squared)])])
        result.add_table_params(self, alpha=alpha, use_t=False)
        return result


def fit_mnl(data, observation_id_col, alternative_id_col, choice_col,
            specification, names=None, initial_values=None):
    """Fit the supported PyLogit-style multinomial logit model."""
    return MNLResults(
        data, observation_id_col, alternative_id_col, choice_col,
        specification, names, initial_values)
