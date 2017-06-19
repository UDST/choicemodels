from __future__ import print_function

import logging

import numpy as np
import pandas as pd
import scipy.optimize

from .tools import pmat
from .tools.pmat import PMAT

from patsy import dmatrix
from urbansim.utils.logutil import log_start_finish


"""
#####################
NEW CLASS DEFINITIONS
#####################

We're refactoring the estimation code to have a more generic interface.

Plan: define a class structure, move functions in one by one, and then do the same for 
interaction.py and pmat.py. Use as light a touch as possible at first, and then take a  
second pass later. This should make it easier to update documentation and tests.

"""


class MultinomialLogit(object):
    """
    A class with methods for estimating multinomial logit discrete choice models.
    
    This is based on the UrbanSim MNL codebase, and functionality from PyLogit is not yet 
    integrated. So for now, each choice scenario must have the same number of alternatives
    (although these can be sampled from a larger set), and the same utility equation
    must be used for all the alternatives.
    
    The utility equation can include attributes of the choosers and of the alternatives. 
    Attributes of a particular alternative may vary for different choosers (distance, for
    example), but the user must set this up manually in the input data.
    
    The input data needs to be in "long" format, with one row for each combination of 
    chooser and alternative. (Sampling of alternatives should be carried out before data 
    is passed to this class.)
    
    Note that prediction methods are in a separate class: MultinomialLogitResults().
    
    Parameters
    ----------
    
    data : pandas.DataFrame
        A table of estimation data in "long" format, with one row for each combination of 
        chooser and alternative. Column labeling must be consistent with the
        'model_expression'. May include extra columns. The table must be sorted such that
        each chooser's alternatives are in contiguous rows.
        
        [TO DO: Can we enforce latter requirement in the code? Tradeoff is that it would
        require additional input (chooser id and alternative id columns).]
    
    choice : str, 1D array, 2D array
        An indication of which alternative has been chosen in each scenario. This can take
        the form of (a) a 1D binary array of same length and order as the 'data' table,
        (b) the name of such a column in the 'data' table, or (c) a 2D binary array with a 
        row for each chooser and a column for each alternative. The column ordering for
        alternatives must be the same as their row ordering in the 'data' table. 
        
        [ONLY FINAL OPTION HAS BEEN IMPLEMENTED SO FAR]
    
    numalts : int
        Number of alternatives in each choice scenario.
        
        [TO DO: is there a better or more flexible way to get this information? For
        example, if we required a chooser ID we could infer the number of alternatives,
        and it would also match PyLogit better.]
        
    model_expression: str, iterable, or dict [CONFIRM]
        A patsy model expression, containing only a right-hand side.
    
    weights : 1D array, optional
        Estimation weights. [TK]
    
    """
    def __init__(self, data, choice, numalts, model_expression, weights=None):
        self.data = data
        self.choice = choice
        self.numalts = numalts
        self.model_expression = model_expression
        self.weights = weights
        return
        
    def _validate_input_data(self):
        return

    def fit(self):
        """
        [TO DO: implement optional parameters]
        
        Parameters
        ----------      
        GPU : bool, optional
            GPU acceleration.       
        coefrange : tuple of floats, optional
            Limits to which coefficients are held, in format (min, max). 
        beta : 1D array, optional
            Initial values for the coefficients.
        
        Returns
        -------
        MultinomialLogitResults() object.
        
        """
        model_design = dmatrix(self.model_expression, data=self.data, 
                               return_type='dataframe').as_matrix()
    
        mnl_params = {'data': model_design,
                      'chosen': self.choice,
                      'numalts': self.numalts}
                      
        log_likelihoods, fit_parameters = mnl_estimate(**mnl_params)

        return MultinomialLogitResults(log_likelihoods, fit_parameters)


class MultinomialLogitResults(object):
    """
        
    """
    def __init__(self, log_likelihoods, fit_parameters):
        self.log_likelihoods = log_likelihoods
        self.fit_parameters = fit_parameters
        return
    
    def __repr__(self):
        return
    
    def __str__(self):
        return self.log_likelihoods.__str__() + self.fit_parameters.__str__()



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
