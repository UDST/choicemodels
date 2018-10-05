"""
Tests for the simulation codebase.

"""
from __future__ import division

import numpy as np
import pandas as pd
import pytest

import choicemodels


# TO DO - could we set a random seed and then verify that monte_carlo_choices() provides
# the same output as np.random.choice()?

def build_data(num_obs, num_alts):
    """
    Build a simulated list of scenarios, alternatives, and probabilities
    
    """
    obs = np.repeat(np.arange(num_obs), num_alts)
    alts = np.random.randint(0, num_alts*10, size=num_obs*num_alts)

    weights = np.random.rand(num_alts, num_obs)
    probs = weights / weights.sum(axis=0)
    probslist = probs.flatten(order='F')

    data = pd.DataFrame({'oid': obs, 'aid': alts, 'probs': probslist})
    data = data.set_index(['oid','aid']).probs
    return data


def test_monte_carlo_choices():
    """
    Test simulation of choices without capacity constraints. This test just verifies that
    the code runs, using a fairly large synthetic dataset.
    
    """
    data = build_data(1000, 100)
    choicemodels.tools.monte_carlo_choices(data)


def test_simulation_accuracy():
    """
    This test checks that the simulation tool is generating choices that match the 
    provided probabilities. 
    
    """
    data = build_data(5,3)
    
    # Get values associated with an arbitrary row
    r = np.random.randint(0, 15, 1)
    row = pd.DataFrame(data).reset_index().iloc[r]
    oid = int(row.oid)
    aid = int(row.aid)
    prob = float(pd.DataFrame(data).query('oid=='+str(oid)+' & aid=='+str(aid)).sum())

    n = 1000
    count = 0
    for i in range(n):
        choices = choicemodels.tools.monte_carlo_choices(data)
        if (choices.loc[oid] == aid):
            count += 1

    assert(count/n > prob-0.1)
    assert(count/n < prob+0.1)


@pytest.fixture
def obs():
    d1 = {'oid': np.arange(50), 
          'obsval': np.random.random(50),
          'choice': np.random.choice(np.arange(60), size=50)}
    return pd.DataFrame(d1).set_index('oid')


@pytest.fixture
def alts():
    d2 = {'aid': np.arange(60), 
          'altval': np.random.random(60)}
    return pd.DataFrame(d2).set_index('aid')


@pytest.fixture
def fitted_model(obs, alts):
    mct = choicemodels.tools.MergedChoiceTable(obs, alts, 'choice', sample_size=5)
    m = choicemodels.MultinomialLogit(mct, model_expression='obsval + altval - 1')
    return m.fit()


def test_iterative_lottery_choices(obs, alts, fitted_model):
    """
    """
    def mct(obs, alts):
        return choicemodels.tools.MergedChoiceTable(obs, alts, sample_size=10)
    
    def probs(mct):
        return fitted_model.probabilities(mct)
    
    choices = choicemodels.tools.iterative_lottery_choices(obs, alts, mct, probs)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    