"""
These are tests for the refactored choicemodels MNL codebase.

"""

import numpy as np
import pandas as pd
import pytest

from pandas.testing import assert_frame_equal
from patsy import dmatrix

from choicemodels import MultinomialLogit
from choicemodels.tools import MergedChoiceTable


@pytest.fixture
def obs():
    d1 = {'oid': np.arange(100), 
          'obsval': np.random.random(100),
          'choice': np.random.choice(np.arange(5), size=100)}
    return pd.DataFrame(d1).set_index('oid')

@pytest.fixture
def alts():
    d2 = {'aid': np.arange(5), 
          'altval': np.random.random(5)}
    return pd.DataFrame(d2).set_index('aid')


def test_mnl(obs, alts):
    """
    Confirm that MNL estimation runs, using the native estimator.
    
    """
    model_expression = 'obsval + altval - 1'
    mct = MergedChoiceTable(obs, alts, 'choice')
    m = MultinomialLogit(mct, model_expression)
    print(m.fit())
    

def test_mnl_estimation(obs, alts):
    """
    Confirm that estimated params from the new interface match urbansim.urbanchoice.
    Only runs if the urbansim package has been installed.
    
    """
    try:
        from urbansim.urbanchoice.mnl import mnl_estimate
    except:
        print("Comparison of MNL estimation results skipped because urbansim is not installed")
        return

    model_expression = 'obsval + altval - 1'
    mct = MergedChoiceTable(obs, alts, 'choice')
    
    # new interface
    m = MultinomialLogit(mct, model_expression)
    r = m.fit().get_raw_results()
    
    # old interface
    dm = dmatrix(model_expression, mct.to_frame())
    chosen = np.reshape(mct.to_frame()[mct.choice_col].values, (100, 5))
    log_lik, fit = mnl_estimate(np.array(dm), chosen, numalts=5)
    
    for k,v in log_lik.items():
        assert(v == pytest.approx(r['log_likelihood'][k], 0.00001))
    
    assert_frame_equal(fit, r['fit_parameters'][['Coefficient', 'Std. Error', 'T-Score']])


def test_mnl_prediction(obs, alts):
    """
    Confirm that fitted probabilities in the new codebase match urbansim.urbanchoice.
    Only runs if the urbansim package has been installed.
    
    """
    try:
        from urbansim.urbanchoice.mnl import mnl_simulate
    except:
        print("Comparison of MNL simulation results skipped because urbansim is not installed")
        return

    # produce a fitted model
    mct = MergedChoiceTable(obs, alts, 'choice', 5)
    m = MultinomialLogit(mct, model_expression='obsval + altval - 1')
    results = m.fit()
    
    # get predicted probabilities using choicemodels
    probs1 = results.probabilities(mct)
    
    # compare to probabilities from urbansim.urbanchoice
    dm = dmatrix(results.model_expression, data=mct.to_frame(), return_type='dataframe')

    probs = mnl_simulate(data=dm, coeff=results.fitted_parameters,
                         numalts=mct.sample_size, returnprobs=True)

    df = mct.to_frame()
    df['prob'] = probs.flatten()
    probs2 = df.prob
    
    pd.testing.assert_series_equal(probs1, probs2)