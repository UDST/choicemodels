"""
Tests for the simulation codebase.

"""
from __future__ import division

import functools
import numpy as np
import pandas as pd
import pytest
import multiprocessing

from choicemodels import MultinomialLogit
from choicemodels.tools import (iterative_lottery_choices, monte_carlo_choices,
        MergedChoiceTable, parallel_lottery_choices)


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
    monte_carlo_choices(data)


def test_simulation_accuracy():
    """
    This test checks that the simulation tool is generating choices that match the 
    provided probabilities. 
    
    """
    data = build_data(5,3)
    
    # Get values associated with an arbitrary row
    r = np.random.randint(0, 15)
    row = pd.DataFrame(data).reset_index().iloc[r]
    oid = int(row.oid)
    aid = int(row.aid)
    prob = float(pd.DataFrame(data).query('oid=='+str(oid)+' & aid=='+str(aid)).probs.sum())

    n = 1000
    count = 0
    for i in range(n):
        choices = monte_carlo_choices(data)
        if (choices.loc[oid] == aid):
            count += 1

    assert(count/n > prob-0.1)
    assert(count/n < prob+0.1)


# CHOICE SIMULATION WITH CAPACITY CONSTRAINTS

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
    mct = MergedChoiceTable(obs, alts, 'choice', sample_size=5)
    m = MultinomialLogit(mct, model_expression='obsval + altval - 1')
    return m.fit()

# The callables passed to parallel_lottery_choices() are sent to worker processes,
# so they must be picklable: module-level functions rather than closures.

def _sample_mct(obs, alts, intx_ops=None):
    return MergedChoiceTable(obs, alts, sample_size=10)

def _failing_probs(mct):
    raise ValueError("probabilities cannot be computed")


def _sample_mct_strict(obs, alts, intx_ops=None):
    # The default sampler happens to tolerate an empty alternatives table, but the
    # callables in downstream models generally do not, so a lottery must stop before
    # it runs out of alternatives rather than rely on the callable coping.
    if len(alts) == 0:
        raise ValueError("no alternatives left to sample")
    return MergedChoiceTable(obs, alts, sample_size=10)

def _predict_probs(model, mct):
    return model.probabilities(mct)

@pytest.fixture
def mct(obs, alts):
    return _sample_mct

@pytest.fixture
def probs(fitted_model, mct):
    return functools.partial(_predict_probs, fitted_model)


def test_iterative_lottery_choices(obs, alts, mct, probs):
    """
    Test that iterative lottery choices can run.

    """
    iterative_lottery_choices(obs, alts, mct, probs)


def test_input_safety(obs, alts, mct, probs):
    """
    Confirm that original copies of the input dataframes are not modified.
    
    """
    orig_obs = obs.copy()
    orig_alts = alts.copy()
    iterative_lottery_choices(obs, alts, mct, probs)
    pd.testing.assert_frame_equal(orig_obs, obs)
    pd.testing.assert_frame_equal(orig_alts, alts)


def test_index_name_retention(obs, alts, mct, probs):
    """
    Confirm retention of index names.
    
    """
    choices = iterative_lottery_choices(obs, alts, mct, probs)
    assert(choices.index.name == obs.index.name)
    assert(choices.name == alts.index.name)
    # TO DO - check for this in the monte carlo choices too
    

def test_unique_choices(obs, alts, mct, probs):
    """
    Confirm unique choices when there's an implicit capacity of 1.
    
    """
    choices = iterative_lottery_choices(obs, alts, mct, probs)
    assert len(choices) == len(choices.unique())


def test_count_capacity(obs, alts, mct, probs):
    """
    Confirm count-based capacity constraints are respected.
    
    """
    alts['capacity'] = np.random.choice([1,2,3], size=len(alts))
    choices = iterative_lottery_choices(obs, alts, mct, probs, alt_capacity='capacity')
    
    placed = pd.DataFrame(choices).groupby('aid').size().rename('placed')
    df = pd.DataFrame(alts.capacity).join(placed, on='aid').fillna(0)
        
    assert(all(df.placed.le(df.capacity)))

    
def test_size_capacity(obs, alts, mct, probs):
    """
    Confirm size-based capacity constraints are respected.
    
    """
    alts['capacity'] = np.random.choice([1,2,3], size=len(alts))
    obs['size'] = np.random.choice([1,2], size=len(obs))
    choices = iterative_lottery_choices(obs, alts, mct, probs, alt_capacity='capacity',
                                        chooser_size='size')
    
    choice_df = pd.DataFrame(choices).join(obs['size'], on='oid')
    placed = choice_df.groupby('aid')['size'].sum().rename('placed')
    df = pd.DataFrame(alts.capacity).join(placed, on='aid').fillna(0)
        
    assert(all(df.placed.le(df.capacity)))

    
def test_insufficient_capacity(obs, alts, mct, probs):
    """
    Confirm that choices are simulated even if there is insufficient overall capacity.
    
    """
    alts = alts.iloc[:30].copy()
    choices = iterative_lottery_choices(obs, alts, mct, probs)
    assert len(choices) > 0
    

def test_exhausted_alternatives(obs, alts, probs):
    """
    Confirm that the lottery stops cleanly when every alternative fills up before the
    choosers run out, without asking for a choice table from an empty alternatives table
    (PR #75).

    """
    alts = alts.iloc[:5].copy()  # 50 choosers, 5 alternatives with capacity 1
    choices = iterative_lottery_choices(obs, alts, _sample_mct_strict, probs)

    assert len(choices) == len(alts)
    assert sorted(choices.values) == alts.index.tolist()


def test_exhausted_alternatives_in_parallel(obs, alts, probs):
    """
    The parallel lottery runs the same check in each worker.

    """
    alts = alts.iloc[:5].copy()
    choices = parallel_lottery_choices(
        obs, alts, _sample_mct_strict, probs, chooser_batch_size=25)

    assert len(choices) == len(alts)
    assert sorted(choices.values) == alts.index.tolist()


def test_chooser_priority(obs, alts, mct, probs):
    """
    Confirm that chooser priority is randomized.
    
    """
    choices = iterative_lottery_choices(obs, alts, mct, probs)
    assert (choices.index.values[:5].tolist != [0, 1, 2, 3, 4])
    
    
def test_max_iter(obs, alts, mct, probs):
    """
    Confirm that max_iter param will prevent infinite loop.
    
    """
    obs['size'] = 2  # (alts have capacity of 1)
    iterative_lottery_choices(obs, alts, mct, probs, chooser_size='size', max_iter=5)


def test_capacity_break(obs, alts, mct, probs):
    """
    Confirm that if alts[capacity].max() < choosers[size].min() will prevent infinite loop.

    """
    obs['size'] = 2
    alts['capacity'] = np.random.choice([3,5], size=len(alts)) # alt capacity left but not enough to host one obs
    iterative_lottery_choices(obs, alts, mct, probs,
                              chooser_size='size', alt_capacity='capacity')


def test_parallel_lottery_choices(obs, alts, mct, probs):
    """
    Test that parallel lottery choices can run, that there aren't any duplicate
    choices, and that every alternative gets filled when they are the binding
    constraint (40 alternatives with capacity 1, 50 choosers).
    
    """
    alts = alts.iloc[:40].copy()
    num_cpus = multiprocessing.cpu_count()
    batch_size = int(np.ceil(len(obs) / num_cpus))
    choices = parallel_lottery_choices(
        obs, alts, mct, probs, chooser_batch_size=batch_size)
    assert not choices.duplicated().any()
    assert sorted(choices.values) == alts.index.tolist()


def test_parallel_lottery_choices_default_batch_size(obs, alts, mct, probs):
    """
    Confirm that omitting chooser_batch_size processes all the choosers in one batch.

    """
    choices = parallel_lottery_choices(obs, alts, mct, probs)
    assert len(choices) == len(obs)
    assert not choices.duplicated().any()


def test_parallel_lottery_choices_rejects_unsupported_ids(obs, alts, mct, probs):
    """
    Alternative ids are exchanged between workers through a shared integer array, so
    they must be non-negative integers; anything else is rejected up front.

    """
    negative = alts.iloc[:5].copy()
    negative.index = pd.Index([-1, 0, 1, 2, 3], name='aid')
    with pytest.raises(ValueError, match="non-negative"):
        parallel_lottery_choices(obs, negative, mct, probs)

    strings = alts.iloc[:5].copy()
    strings.index = pd.Index(list('abcde'), name='aid')
    with pytest.raises(ValueError, match="integer alternative ids"):
        parallel_lottery_choices(obs, strings, mct, probs)


def test_parallel_lottery_choices_surfaces_worker_errors(obs, alts, mct):
    """
    An error inside a worker process must raise in the caller rather than silently
    returning a short result.

    """
    with pytest.raises(RuntimeError, match="exited with an error"):
        parallel_lottery_choices(obs, alts, mct, _failing_probs, chooser_batch_size=25)
