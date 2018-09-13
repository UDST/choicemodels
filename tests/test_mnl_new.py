"""
These are tests for the refactored choicemodels codebase.

"""

import numpy as np
import pandas as pd
import pytest

import choicemodels


d1 = {'oid': np.arange(100), 
      'obsval': np.random.random(100),
      'choice': np.random.choice(np.arange(5), size=100)}

d2 = {'aid': np.arange(5), 
      'altval': np.random.random(5)}

obs = pd.DataFrame(d1).set_index('oid')
alts = pd.DataFrame(d2).set_index('aid')


def test_prediction():
    import patsy
    from urbansim.urbanchoice import mnl

    # produce a fitted model
    mct = choicemodels.tools.MergedChoiceTable(obs, alts, 'choice', 5)
    m = choicemodels.MultinomialLogit(mct, model_expression='obsval + altval - 1')
    results = m.fit()
    
    # get predicted probabilities using choicemodels
    probs1 = results.probabilities(mct)
    
    # compare to probabilities from urbansim.urbanchoice
    dm = patsy.dmatrix(results.model_expression, data=mct.to_frame(),
                       return_type='dataframe')

    probs = mnl.mnl_simulate(data=dm, coeff=results.fitted_parameters,
                             numalts=mct.sample_size, returnprobs=True)

    df = mct.to_frame()
    df['prob'] = probs.flatten()
    probs2 = df.prob
    
    pd.testing.assert_series_equal(probs1, probs2)