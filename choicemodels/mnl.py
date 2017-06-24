from __future__ import print_function

import logging

import numpy as np
import pandas as pd
import pylogit
import scipy.optimize

from .tools import pmat
from .tools.pmat import PMAT

from collections import OrderedDict
from patsy import dmatrix
from prettytable import PrettyTable
from urbansim.utils.logutil import log_start_finish


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
       'model_expression' in Patsy format. 
       
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
    
    [TO DO: comparison of the estimation engines]
    
    Note that prediction methods are in a separate class: see MultinomialLogitResults().
        
    Parameters
    ----------
    
    data : pandas.DataFrame
        A table of estimation data in "long" format, with one row for each combination of 
        chooser and alternative. Column labeling must be consistent with the
        'model_expression'. May include extra columns. 
   
    observation_id_col : str
        Name of column containing the observation id. This should uniquely identify each
        distinct choice scenario. 
        
    choice_col : str
        Name of column containing an indication of which alternative has been chosen in 
        each scenario. Values should evaluate as binary: 1/0, True/False, etc.
    
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

    model_labels : PyLogit 'names', optional
        If the model expression is a PyLogit OrderedDict, you can provide a corresponding 
        OrderedDict of labels. See here:
        https://github.com/timothyb0912/pylogit/blob/master/pylogit/pylogit.py#L151-L165

    alternative_id_col : str, optional
        Name of column containing the alternative id. This is only required if the model 
        expression varies for different alternatives.
    
    initial_coefs : float, list, or 1D array, optional
        Initial coefficients (beta values) to begin the optimization process with. Provide 
        a single value for all coefficients, or an array containing a value for each 
        one being estimated. If None, initial coefficients will be 0.
        
    weights : 1D array, optional
        Estimation weights. [TO DO: implement the pass-through]
    
    """
    def __init__(self, data, observation_id_col, choice_col, model_expression, 
                 model_labels=None, alternative_id_col=None, initial_coefs=None, 
                 weights=None):
        self._data = data
        self._observation_id_col = observation_id_col
        self._alternative_id_col = alternative_id_col
        self._choice_col = choice_col
        self._model_expression = model_expression
        self._model_labels = model_labels
        self._initial_coefs = initial_coefs
        self._weights = weights

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
            self._numobs = self._data[[self._observation_id_col]].\
                                    drop_duplicates().shape[0]
            print(self._numobs)
            self._numalts = self._data.shape[0] / self._numobs
            print(self._numalts)
            
            # TO DO: parse initial coefs
        
        self._validate_input_data()
        return
        
    def _validate_input_data(self):
        """
        [TO DO for ChoiceModels engine:
         - verify number of alternatives consistent for each chooser
         - verify each chooser's alternatives in contiguous rows
         - verify chosen alternative listed first]
                
        """
        self._data = self._data.sort_values(by = [self._observation_id_col,
                                                  self._choice_col],
                                            ascending = [True, False])
        return

    def fit(self):
        """
        Fit the model using maximum likelihood estimation. Uses either the ChoiceModels
        or PyLogit estimation engine as appropriate.
        
        TO DO: should we implement parameters here, or take them all in the constructor?
        
        Parameters
        ----------      
        GPU : bool, optional
            GPU acceleration.       
        coefrange : tuple of floats, optional
            Limits to which coefficients are held, in format (min, max). 
        initial_values : 1D array, optional
            Initial values for the coefficients.
        
        Returns
        -------
        MultinomialLogitResults() object.
        
        """
        if (self._estimation_engine == 'PyLogit'):
            
            m = pylogit.create_choice_model(data = self._data,
                                            obs_id_col = self._observation_id_col,
                                            alt_id_col = self._alternative_id_col,
                                            choice_col = self._choice_col,
                                            specification = self._model_expression,
                                            names = self._model_labels,
                                            model_type = 'MNL')
            
            m.fit_mle(init_vals = self._initial_coefs)
            results = m
            
        elif (self._estimation_engine == 'ChoiceModels'):

            model_design = dmatrix(self._model_expression, data=self._data, 
                                   return_type='dataframe').as_matrix()
    
            # generate 2D array from choice column, for mnl_estimate()
            chosen = np.reshape(self._data[[self._choice_col]].as_matrix(), 
                                (self._numobs, self._numalts))
                
            log_lik, fit = mnl_estimate(model_design, chosen, self._numalts)
            results = MultinomialLogitResults(log_likelihoods = log_lik,
                                              fit_parameters = fit,
                                              estimation_engine = self._estimation_engine)

        return results
        
    
    @property
    def estimation_engine(self):
        """
        
        """
        return self._estimation_engine


class MultinomialLogitResults(object):
    """
    This is a work in progress. The rationale for having a separate results class is that 
    users don't have to keep track of which methods they're allowed to run depending on 
    the status of the object. An estimation object is always ready to estimate, and a 
    results object is always ready to report the results or predict. 
    
    Anticipated functionality of the results class:
    - store estimation results, test statistics, and other metadata for a single model
    - report these in a standard estimation results table
    - provide access to individual values as needed
    - write results to a human-readable text file, and read them back in
    - prediction?? maybe this should be separate and take a results object as input
    
    Most of this functionality can be inherited from a generic results class.
    
    TO DO:
    - can PyLogit results easily be pulled out from a model object?
    - can the PyLogit print function be called on its own?
    - if so, we can store metadata in a unified way and have all output look the same
    
    """
    def __init__(self, log_likelihoods, fit_parameters, estimation_engine):
        self._log_likelihoods = log_likelihoods
        self._fit_parameters = fit_parameters
        self._estimation_engine = estimation_engine
        return
    
    def __repr__(self):
        self.report_fit()
    
    def __str__(self):
        return self._log_likelihoods.__str__() + self._fit_parameters.__str__()

    def report_fit(self):
        """
        Print a report of the fit results. Code from 
        urbansim.models.MNLDiscreteChoiceModel().
        
        """
        print('Null Log-liklihood: {0:.3f}'.format(
            self._log_likelihoods['null']))
        print('Log-liklihood at convergence: {0:.3f}'.format(
            self._log_likelihoods['convergence']))
        print('Log-liklihood Ratio: {0:.3f}\n'.format(
            self._log_likelihoods['ratio']))

        tbl = PrettyTable(
            ['Component', ])
        tbl = PrettyTable()

        tbl.add_column('Component', self._fit_parameters.index.values)
        for col in ('Coefficient', 'Std. Error', 'T-Score'):
            tbl.add_column(col, self._fit_parameters[col].values)

        tbl.align['Component'] = 'l'
        tbl.float_format = '.3'

        print(tbl)


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

    exponentiated_utility = utilities.exp(inplace=True)
    if clamp:
        exponentiated_utility.inftoval(1e20)
    if clamp:
        exponentiated_utility.clamptomin(1e-300)
    sum_exponentiated_utility = exponentiated_utility.sum(axis=0)
    probs = exponentiated_utility.divide_by_row(
        sum_exponentiated_utility, inplace=True)
    if clamp:
        probs.nantoval(1e-300)
    if clamp:
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


def mnl_simulate(data, coeff, numalts, GPU=False, returnprobs=True):
    """
    Get the probabilities for each chooser choosing between `numalts`
    alternatives.

    Parameters
    ----------
    data : 2D array
        The data are expected to be in "long" form where each row is for
        one alternative. Alternatives are in groups of `numalts` rows per
        choosers. Alternatives must be in the same order for each chooser.
    coeff : 1D array
        The model coefficients corresponding to each column in `data`.
    numalts : int
        The number of alternatives available to each chooser.
    GPU : bool, optional
    returnprobs : bool, optional
        If True, return the probabilities for each chooser/alternative instead
        of actual choices.

    Returns
    -------
    probs or choices: 2D array
        If `returnprobs` is True the probabilities are a 2D array with a
        row for each chooser and columns for each alternative.

    """
    logger.debug(
        'start: MNL simulation with len(data)={} and numalts={}'.format(
            len(data), numalts))
    atype = 'numpy' if not GPU else 'cuda'

    data = np.transpose(data)
    coeff = np.reshape(np.array(coeff), (1, len(coeff)))

    data, coeff = PMAT(data, atype), PMAT(coeff, atype)

    probs = mnl_probs(data, coeff, numalts)

    if returnprobs:
        return np.transpose(probs.get_mat())

    # convert to cpu from here on - gpu doesn't currently support these ops
    if probs.typ == 'cuda':
        probs = PMAT(probs.get_mat())

    probs = probs.cumsum(axis=0)
    r = pmat.random(probs.size() // numalts)
    choices = probs.subtract(r, inplace=True).firstpositive(axis=0)

    logger.debug('finish: MNL simulation')
    return choices.get_mat()


def mnl_estimate(data, chosen, numalts, GPU=False, coeffrange=(-3, 3),
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
        the model fit.
    fit_parameters : pandas.DataFrame
        Table of fit parameters with columns 'Coefficient', 'Std. Error',
        'T-Score'. Each row corresponds to a column in `data` and are given
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

    with log_start_finish('scipy optimization for MNL fit', logger):
        args = (data, chosen, numalts, weights, lcgrad)
        bfgs_result = scipy.optimize.fmin_l_bfgs_b(mnl_loglik,
                                                   beta,
                                                   args=args,
                                                   fprime=None,
                                                   factr=10,
                                                   approx_grad=False,
                                                   bounds=bounds
                                                   )
    beta = bfgs_result[0]
    stderr = mnl_loglik(
        beta, data, chosen, numalts, weights, stderr=1, lcgrad=lcgrad)

    l0beta = np.zeros(numvars)
    l0 = -1 * mnl_loglik(l0beta, *args)[0]
    l1 = -1 * mnl_loglik(beta, *args)[0]

    log_likelihood = {
        'null': float(l0[0][0]),
        'convergence': float(l1[0][0]),
        'ratio': float((1 - (l1 / l0))[0][0])
    }

    fit_parameters = pd.DataFrame({
        'Coefficient': beta,
        'Std. Error': stderr,
        'T-Score': beta / stderr})

    logger.debug('finish: MNL fit')
    return log_likelihood, fit_parameters
