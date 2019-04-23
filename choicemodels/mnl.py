from __future__ import print_function

import datetime
import logging
from collections import OrderedDict

import numpy as np
import pandas as pd
import pylogit
import scipy.optimize
import scipy.stats
from patsy import dmatrix
from statsmodels.iolib.table import SimpleTable

from .tools import MergedChoiceTable
from .tools import pmat
from .tools.pmat import PMAT


"""
#####################
NEW CLASS DEFINITIONS
#####################

"""


class MultinomialLogit(object):
    """
    A class with methods for estimating multinomial logit discrete choice models. Each
    observation is a choice scenario in which a chooser selects one alternative from a
    choice set of two or more. The fitted parameters represent a joint optimization of
    utility expressions that explains observed choices based on attributes of the
    alternatives and of the choosers.

    The input data needs to be in "long" format, with one row for each combination of
    chooser and alternative. Columns contain relevant attributes and identifiers. (If the
    choice sets are large, sampling of alternatives should be carried out before data is
    passed to this class.)

    The class constructor supports two use cases:

    1. The first use case is simpler and requires fewer inputs. Each choice scenario must
       have the same number of alternatives, and each alternative must have the same model
       expression (utility equation). This is typical when the alternatives are relatively
       numerous and homogenous, for example with travel destination choice or household
       location choice.

       The following parameters are required: 'data', 'observation_id_col', 'choice_col',
       'model_expression' in Patsy format. If data is provided as a MergedChoiceTable,
       the observation id and choice column names can be read directly from its metadata.

       To fit this type of model, ChoiceModels will use its own estimation engine adapted
       from the UrbanSim MNL codebase.

       Migration from 'urbansim.urbanchoice': Note that these requirements differ from
       the old UrbanSim codebase in a couple of ways. (1) The chosen alternatives need to
       be indicated in a column of the estimation data table instead of in a separate
       matrix, and (2) in lieu of indicating the number of alternatives in each choice
       set, the estimation data table should include an observation id column. These
       changes make the API more consistent with other use cases. See the
       MergedChoiceTable() class for tools and code examples to help with migration.

    2. The second use case is more flexible. Choice scenarios can have varying numbers of
       alternatives, and the model expression (utility equation) can be different for
       distinct alternatives. This is typical when there is a small number of alternatives
       whose salient characteristics vary, for example with travel mode choice.

       The following parameters are required: 'data', 'observation_id_col',
       'alternative_id_col', 'choice_col', 'model_expression' in PyLogit format,
       'model_labels' in PyLogit format (optional).

       To fit this type of model, ChoiceModels will use the PyLogit estimation engine.

    With either use case, the model expression can include attributes of both the choosers
    and the alternatives. Attributes of a particular alternative may vary for different
    choosers (distance, for example), but this must be set up manually in the input data.

    Note that prediction methods are in a separate class: see MultinomialLogitResults().

    Parameters
    ----------

    data : pd.DataFrame or choicemodels.tools.MergedChoiceTable
        A table of estimation data in "long" format, with one row for each combination of
        chooser and alternative. Column labeling must be consistent with the
        'model_expression'. May include extra columns.

    model_expression : Patsy 'formula-like' or PyLogit 'specification'
        For the simpler use case where each choice scenario has the same number of
        alternatives and each alternative has the same model expression, this should be a
        Patsy formula representing the right-hand side of the single model expression.
        This can be a string or a number of other data types. See here:
        https://patsy.readthedocs.io/en/v0.1.0/API-reference.html#patsy.dmatrix

        For the more flexible use case where choice scenarios have varying numbers of
        alternatives or the model expessions vary, this should be a PyLogit OrderedDict
        model specification. See here:
        https://github.com/timothyb0912/pylogit/blob/master/pylogit/pylogit.py#L116-L130

    observation_id_col : str, optional
        Name of column or index containing the observation id. This should uniquely 
        identify each distinct choice scenario. Not required if data is passed as a 
        MergedChoiceTable.

    choice_col : str, optional
        Name of column containing an indication of which alternative has been chosen in
        each scenario. Values should evaluate as binary: 1/0, True/False, etc. Not
        required if data is passed as a MergedChoiceTable.

    model_labels : PyLogit 'names', optional
        If the model expression is a PyLogit OrderedDict, you can provide a corresponding
        OrderedDict of labels. See here:
        https://github.com/timothyb0912/pylogit/blob/master/pylogit/pylogit.py#L151-L165

    alternative_id_col : str, optional
        Name of column or index containing the alternative id. This is only required if 
        the model expression varies for different alternatives. Not required if data is 
        passed as a MergedChoiceTable.

    initial_coefs : numeric or list-like of numerics, optional
        Initial coefficients (beta values) to begin the optimization process with. Provide
        a single value for all coefficients, or an array containing a value for each
        one being estimated. If None, initial coefficients will be 0.

    weights : 1D array, optional
        NOT YET IMPLEMENTED - Estimation weights.

    """
    def __init__(self, data, model_expression, observation_id_col=None, choice_col=None,
                 model_labels=None, alternative_id_col=None, initial_coefs=None,
                 weights=None):
        self._data = data
        self._model_expression = model_expression
        self._observation_id_col = observation_id_col
        self._alternative_id_col = alternative_id_col
        self._choice_col = choice_col
        self._model_labels = model_labels
        self._initial_coefs = initial_coefs
        self._weights = weights

        if isinstance(self._data, MergedChoiceTable):
            self._df = self._data.to_frame()
            self._observation_id_col = self._data.observation_id_col
            self._alternative_id_col = self._data.alternative_id_col
            self._choice_col = self._data.choice_col
        
        else:
            self._df = self._data
        
        if isinstance(self._model_expression, OrderedDict):
            self._estimation_engine = 'PyLogit'

            # parse initial_coefs
            if isinstance(self._initial_coefs, np.ndarray):
                pass
            elif isinstance(self._initial_coefs, list):
                self._initial_coefs = np.array(self._initial_coefs)
            elif (self._initial_coefs == None):
                self._initial_coefs = np.zeros(len(self._model_expression))
            else:
                self._initial_coefs = np.repeat(self._initial_coefs,
                                                len(self._model_expression))

        else:
            self._estimation_engine = 'ChoiceModels'
            self._numobs = self._df.reset_index()[[self._observation_id_col]].\
                                    drop_duplicates().shape[0]
            self._numalts = self._df.shape[0] // self._numobs

        return

    
    def fit(self):
        """
        Fit the model using maximum likelihood estimation. Uses either the ChoiceModels
        or PyLogit estimation engine as appropriate.

        Returns
        -------
        MultinomialLogitResults() object.

        """
        if (self._estimation_engine == 'PyLogit'):

            m = pylogit.create_choice_model(data = self._df,
                                            obs_id_col = self._observation_id_col,
                                            alt_id_col = self._alternative_id_col,
                                            choice_col = self._choice_col,
                                            specification = self._model_expression,
                                            names = self._model_labels,
                                            model_type = 'MNL')

            m.fit_mle(init_vals = self._initial_coefs)
            results = MultinomialLogitResults(estimation_engine = self._estimation_engine,
                                              model_expression = self._model_expression,
                                              results = m)

        elif (self._estimation_engine == 'ChoiceModels'):

            dm = dmatrix(self._model_expression, data=self._df)

            chosen = np.reshape(self._df[[self._choice_col]].values,
                                (self._numobs, self._numalts))

            log_lik, fit = mnl_estimate(np.array(dm), chosen, self._numalts)

            result_params = dict(log_likelihood = log_lik,
                                 fit_parameters = fit,
                                 x_names = dm.design_info.column_names)

            results = MultinomialLogitResults(estimation_engine = self._estimation_engine,
                                              model_expression = self._model_expression,
                                              results = result_params)

        return results


    @property
    def estimation_engine(self):
        """
        'ChoiceModels' or 'PyLogit'.

        """
        return self._estimation_engine


class MultinomialLogitResults(object):
    """
    The results class represents a fitted model. It can report the model fit, generate
    choice probabilties, etc.
    
    A full-featured results object is returned by MultinomialLogit.fit(). A results object
    with more limited functionality can also be built directly from fitted parameters and
    a model expression.

    Parameters
    ----------
    model_expression : str or OrderedDict
        Patsy 'formula-like' (str) or PyLogit 'specification' (OrderedDict).
    
    results : dict or object, optional
        Raw results as currently provided by the estimation engine. This should be
        replaced with a more consistent and comprehensive set of inputs.

    fitted_parameters : list of floats, optional
        If not provided, these will be extracted from the raw results.

    estimation_engine : str, optional
        'ChoiceModels' (default) or 'PyLogit'.

    """
    def __init__(self, model_expression, results=None, fitted_parameters=None, 
                 estimation_engine='ChoiceModels'):
        
        if (fitted_parameters is None) & (results is not None):
            if (estimation_engine == 'ChoiceModels'):
                fitted_parameters = results['fit_parameters']['Coefficient'].tolist()

        self.estimation_engine = estimation_engine
        self.model_expression = model_expression
        self.results = results
        self.fitted_parameters = fitted_parameters
        
    
    def __repr__(self):
    	return self.report_fit()

    
    def __str__(self):
        return self.report_fit()

    
    def get_raw_results(self):
        """
        Return the raw results as provided by the estimation engine. Dict or object.

        """
        return self.results

    
    def probabilities(self, data):
        """
        Generate predicted probabilities for a table of choice scenarios, using the fitted
        parameters stored in the results object.
        
        Parameters
        ----------
        data : choicemodels.tools.MergedChoiceTable
            Long-format table of choice scenarios. TO DO - accept other data formats.
        
        Expected class parameters
        -------------------------
        self.model_expression : patsy string
        self.fitted_parameters : list of floats
        
        Returns
        -------
        pandas.Series with indexes matching the input
        
        """
        # TO DO - make sure this handles pylogit case
        
        # TO DO - does MergedChoiceTable guarantee that alternatives for a single scenario
        # are consecutive? seems like a requirement here; should document it
        
        df = data.to_frame()
        numalts = data.sample_size  # TO DO - make this an official MCT param
        
        dm = dmatrix(self.model_expression, data=df)
        
        # utility is sum of data values times fitted betas
        u = np.dot(self.fitted_parameters, np.transpose(dm))
        
        # reshape so axis 0 lists alternatives and axis 1 lists choosers
        u = np.reshape(u, (numalts, u.size // numalts), order='F')
    
        # scale the utilities to make exponentiation easier
        # https://stats.stackexchange.com/questions/304758/softmax-overflow
        u = u - u.max(axis=0)
        
        exponentiated_utility = np.exp(u)
        sum_exponentiated_utility = np.sum(exponentiated_utility, axis=0)
        
        probs = exponentiated_utility / sum_exponentiated_utility
        
        # convert back to ordering of the input data
        probs = probs.flatten(order='F')
        
        df['prob'] = probs  # adds indexes
        return df.prob
    
    
    def report_fit(self):
        """
        Print a report of the model estimation results.

        """
        if (self.estimation_engine == 'PyLogit'):
            output = self.results.get_statsmodels_summary().as_text()

        elif (self.estimation_engine == 'ChoiceModels'):

            # Pull out individual results components
            ll = self.results['log_likelihood']['convergence']
            ll_null = self.results['log_likelihood']['null']

            rho_bar_squared = self.results['log_likelihood']['rho_bar_squared']
            rho_squared = self.results['log_likelihood']['rho_squared']
            df_model = self.results['log_likelihood']['df_model']
            df_resid = self.results['log_likelihood']['df_resid']
            num_obs = self.results['log_likelihood']['num_obs']
            bic = self.results['log_likelihood']['bic']
            aic = self.results['log_likelihood']['aic']

            x_names = self.results['x_names']
            coefs = self.results['fit_parameters']['Coefficient'].tolist()
            std_errs = self.results['fit_parameters']['Std. Error'].tolist()
            t_scores = self.results['fit_parameters']['T-Score'].tolist()
            p_values = self.results['fit_parameters']['P-Values'].tolist()

            def time_now(*args, **kwds):
                now = datetime.datetime.now()
                return now.strftime('%H:%M')

            def date_now(*args, **kwds):
                now = datetime.datetime.now()
                return now.strftime('%Y-%m-%d')

            (header, body) = summary_table(dep_var = 'chosen',
                                           model_name = 'Multinomial Logit',
                                           method = 'Maximum Likelihood',
                                           log_likelihood = ll,
                                           null_log_likelihood = ll_null,
                                           rho_squared = rho_squared,
                                           rho_bar_squared = rho_bar_squared,
                                           df_model = df_model,
                                           df_resid = df_resid,
                                           x_names = x_names,
                                           coefs = coefs,
                                           std_errs = std_errs,
                                           t_scores = t_scores,
                                           p_values = p_values,
                                           bic = bic,
                                           aic = aic,
                                           num_obs = num_obs,
                                           time = time_now(),
                                           date = date_now())

            output = header.as_text() + '\n' + body.as_text()

        return output


def summary_table(title=None, dep_var='', model_name='', method='', date='',
                  time='', aic=None, bic=None, num_obs=None, df_resid=None,
                  df_model=None, rho_squared=None, rho_bar_squared=None,
                  log_likelihood=None, null_log_likelihood=None, x_names=[], coefs=[],
                  std_errs=[], t_scores=[],p_values=[], alpha=None):
    """
    Generate a summary table of estimation results using Statsmodels SimpleTable. Still a
    work in progress.

    SimpleTable is maddening to work with, so it would be nice to find an alternative. It
    would need to support pretty-printing of formatted tables to plaintext and ideally
    also to HTML and Latex.

    At first it looked like we could use Statsmodels's summary table generator directly
    (iolib.summary.Summary), but this requires a Statsmodels results object as input and
    doesn't document which properties are pulled from it. PyLogit reverse engineered this
    for use in get_statsmodels_summary() -- so it's possible, but could be hard to
    maintain in the long run.

    We can't use PyLogit's summary table generator either. It requires a PyLogit
    model class as input, and we can't create one from results parameters. Oh well!

    """
    def fmt(value, format_str):
        # Custom numeric->string formatter that gracefully accepts null values
        return '' if value is None else format_str.format(value)

    if (title is None):
        title = "CHOICEMODELS ESTIMATION RESULTS"

    top_left = [['Dep. Var.:', dep_var],
                ['Model:', model_name],
                ['Method:', method],
                ['Date:', date],
                ['Time:', time],
                ['AIC:', fmt(aic, "{:,.3f}")],
                ['BIC:', fmt(bic, "{:,.3f}")]]

    top_right = [['No. Observations:', fmt(num_obs, "{:,}")],
                 ['Df Residuals:', fmt(df_resid, "{:,}")],
                 ['Df Model:', fmt(df_model, "{:,}")],
                 ['Pseudo R-squ.:', fmt(rho_squared, "{:.3f}")],
                 ['Pseudo R-bar-squ.:', fmt(rho_bar_squared, "{:.3f}")],
                 ['Log-Likelihood:', fmt(log_likelihood, "{:,.3f}")],
                 ['LL-Null:', fmt(null_log_likelihood, "{:,.3f}")]]

    # Zip into a single table (each side needs same number of entries)
    header_cells = [top_left[i] + top_right[i] for i in range(len(top_left))]

    # See end of statsmodels.iolib.table.py for formatting options
    header_fmt = dict(table_dec_below = '',
                      data_aligns = 'lrlr',
                      colwidths = 11,
                      colsep = '   ',
                      empty_cell = '')

    header = SimpleTable(header_cells, title=title, txt_fmt=header_fmt)

    col_labels = ['coef', 'std err', 'z', 'P>|z|', 'Conf. Int.']
    row_labels = x_names

    body_cells = [[fmt(coefs[i], "{:,.4f}"),
                   fmt(std_errs[i], "{:,.3f}"),
                   fmt(t_scores[i], "{:,.3f}"),
                   fmt(p_values[i], "{:,.3f}"),
                   '']  # conf int placeholder
                   for i in range(len(x_names))]

    body_fmt = dict(table_dec_below = '=',
                    header_align = 'r',
                    data_aligns = 'r',
                    colwidths = 7,
                    colsep = '   ')

    body = SimpleTable(body_cells,
                       headers = col_labels,
                       stubs = row_labels,
                       txt_fmt = body_fmt)

    # Ideally we'd want to append these into a single table, but I can't get it to work
    # without completely messing up the formatting..

    return (header, body)


"""
#############################
ORIGINAL FUNCTION DEFINITIONS
#############################

"""


logger = logging.getLogger(__name__)

# right now MNL can only estimate location choice models, where every equation
# is the same
# it might be better to use stats models for a non-location choice problem

# data should be column matrix of dimensions NUMVARS x (NUMALTS*NUMOBVS)
# beta is a row vector of dimensions 1 X NUMVARS


def mnl_probs(data, beta, numalts):
    logging.debug('start: calculate MNL probabilities')
    clamp = data.typ == 'numpy'
    utilities = beta.multiply(data)
    if numalts == 0:
        raise Exception("Number of alternatives is zero")
    utilities.reshape(numalts, utilities.size() // numalts)

    # https://stats.stackexchange.com/questions/304758/softmax-overflow
    utilities = utilities.subtract(utilities.max(0))
    exponentiated_utility = utilities.exp(inplace=True)
    if clamp:
        exponentiated_utility.inftoval(1e20)
        exponentiated_utility.clamptomin(1e-300)
    sum_exponentiated_utility = exponentiated_utility.sum(axis=0)
    probs = exponentiated_utility.divide_by_row(
        sum_exponentiated_utility, inplace=True)
    if clamp:
        probs.nantoval(1e-300)
        probs.clamptomin(1e-300)

    logging.debug('finish: calculate MNL probabilities')
    return probs


def get_hessian(derivative):
    return np.linalg.inv(np.dot(derivative, np.transpose(derivative)))


def get_standard_error(hessian):
    return np.sqrt(np.diagonal(hessian))

# data should be column matrix of dimensions NUMVARS x (NUMALTS*NUMOBVS)
# beta is a row vector of dimensions 1 X NUMVARS


def mnl_loglik(beta, data, chosen, numalts, weights=None, lcgrad=False,
               stderr=0):
    logger.debug('start: calculate MNL log-likelihood')
    numvars = beta.size
    numobs = data.size() // numvars // numalts

    beta = np.reshape(beta, (1, beta.size))
    beta = PMAT(beta, data.typ)

    probs = mnl_probs(data, beta, numalts)

    # lcgrad is the special gradient for the latent class membership model
    if lcgrad:
        assert weights
        gradmat = weights.subtract(probs).reshape(probs.size(), 1)
        gradarr = data.multiply(gradmat)
    else:
        if not weights:
            gradmat = chosen.subtract(probs).reshape(probs.size(), 1)
        else:
            gradmat = chosen.subtract(probs).multiply_by_row(
                weights).reshape(probs.size(), 1)
        gradarr = data.multiply(gradmat)

    if stderr:
        gradmat = data.multiply_by_row(gradmat.reshape(1, gradmat.size()))
        gradmat.reshape(numvars, numalts * numobs)
        return get_standard_error(get_hessian(gradmat.get_mat()))

    chosen.reshape(numalts, numobs)
    if weights is not None:
        if probs.shape() == weights.shape():
            loglik = ((probs.log(inplace=True)
                       .element_multiply(weights, inplace=True)
                       .element_multiply(chosen, inplace=True))
                      .sum(axis=1).sum(axis=0))
        else:
            loglik = ((probs.log(inplace=True)
                       .multiply_by_row(weights, inplace=True)
                       .element_multiply(chosen, inplace=True))
                      .sum(axis=1).sum(axis=0))
    else:
        loglik = (probs.log(inplace=True).element_multiply(
            chosen, inplace=True)).sum(axis=1).sum(axis=0)

    if loglik.typ == 'numpy':
        loglik, gradarr = loglik.get_mat(), gradarr.get_mat().flatten()
    else:
        loglik = loglik.get_mat()[0, 0]
        gradarr = np.reshape(gradarr.get_mat(), (1, gradarr.size()))[0]

    logger.debug('finish: calculate MNL log-likelihood')
    return -1 * loglik, -1 * gradarr


def mnl_estimate(data, chosen, numalts, GPU=False, coeffrange=(-1000, 1000),
                 weights=None, lcgrad=False, beta=None):
    """
    Calculate coefficients of the MNL model.

    Parameters
    ----------
    data : 2D array
        The data are expected to be in "long" form where each row is for
        one alternative. Alternatives are in groups of `numalts` rows per
        choosers. Alternatives must be in the same order for each chooser.
    chosen : 2D array
        This boolean array has a row for each chooser and a column for each
        alternative. The column ordering for alternatives is expected to be
        the same as their row ordering in the `data` array.
        A one (True) indicates which alternative each chooser has chosen.
    numalts : int
        The number of alternatives.
    GPU : bool, optional
    coeffrange : tuple of floats, optional
        Limits of (min, max) to which coefficients are clipped.
    weights : ndarray, optional
    lcgrad : bool, optional
    beta : 1D array, optional
        Any initial guess for the coefficients.

    Returns
    -------
    log_likelihood : dict
        Dictionary of log-likelihood values describing the quality of
        the model fit. The  following keys is entered into `log_likelihood`.
        
        aic : float
            Akaike information criterion for an estimated model. 
            aic =  -2 * log_likelihood + 2 * num_estimated_parameters 
        
        bic : float
            Bayesian information criterion for an estimated model. 
            bic = -2 * log_likelihood + log(num_observations) * num_parameters
        
        num_obs : int
            Number of observations in the model.

        df_model : int
            The number of parameters estimated in this model.

        df_resid : int
            The number of observations minus the number of estimated parameters.

        llnull : float
            Value of the constant-only loglikelihood
        
        ll : float
            Value of the loglikelihood

        rho_squared  : float
            McFadden's pseudo-R-squared. 
            rho_squared = 1.0 - final_log_likelihood / null_log_likelihood

        rho_bar_squared : float
            rho_bar_squared = 1.0 - ((final_log_likelihood - num_est_parameters) /
                             null_log_likelihood)


    fit_parameters : pandas.DataFrame
        Table of fit parameters with columns 'Coefficient', 'Std. Error',
        'T-Score', and 'P-Values. Each row corresponds to a column in `data` and are given
        in the same order as in `data`.

    See Also
    --------
    scipy.optimize.fmin_l_bfgs_b : The optimization routine used.

    """
    logger.debug(
        'start: MNL fit with len(data)={} and numalts={}'.format(
            len(data), numalts))
    atype = 'numpy' if not GPU else 'cuda'

    numvars = data.shape[1]
    numobs = data.shape[0] // numalts

    if chosen is None:
        chosen = np.ones((numobs, numalts))  # used for latent classes

    data = np.transpose(data)
    chosen = np.transpose(chosen)

    data, chosen = PMAT(data, atype), PMAT(chosen, atype)
    if weights is not None:
        weights = PMAT(np.transpose(weights), atype)

    if beta is None:
        beta = np.zeros(numvars)
    bounds = [coeffrange] * numvars

    # scipy optimization for MNL fit
    logger.debug('start: scipy optimization for MNL fit')
    args = (data, chosen, numalts, weights, lcgrad)
    bfgs_result = scipy.optimize.fmin_l_bfgs_b(mnl_loglik,
                                               beta,
                                               args=args,
                                               fprime=None,
                                               factr=10,
                                               approx_grad=False,
                                               bounds=bounds)
    logger.debug('finish: scipy optimization for MNL fit')

    if bfgs_result[2]['warnflag'] > 0:
        logger.warning("mnl did not converge correctly: %s",  bfgs_result)
    
    beta = bfgs_result[0]
    stderr = mnl_loglik(
        beta, data, chosen, numalts, weights, stderr=1, lcgrad=lcgrad)

    l0beta = np.zeros(numvars)
    l0 = -1 * mnl_loglik(l0beta, *args)[0]
    l1 = -1 * mnl_loglik(beta, *args)[0]
    ll = float(l1[0][0])
    ll_null = float(l0[0][0])
    rho_squared = 1.0 - ll / ll_null
    rho_bar_squared = 1.0 - ((ll - len(beta)) /ll_null)
    num_obs = numobs
    df_resid = numobs - len(beta)
    p_values = 2 * scipy.stats.norm.sf(np.abs(beta / stderr))
    bic = -2 * ll + np.log(numobs) * len(beta)
    aic = -2 * ll + 2 * len(beta)

    log_likelihood = {
        'null': float(l0[0][0]),
        'convergence': float(l1[0][0]),
        'ratio': float((1 - (l1 / l0))[0][0]),
        'rho_bar_squared': rho_bar_squared,
        'rho_squared': rho_squared,
        'df_model': len(beta),
        'df_resid': df_resid,
        'num_obs':numobs,
        'bic':bic,
        'aic':aic}

    fit_parameters = pd.DataFrame({
        'Coefficient': beta,
        'Std. Error': stderr,
        'T-Score': beta / stderr,
        'P-Values' : p_values })

    logger.debug('finish: MNL fit')
    return log_likelihood, fit_parameters
