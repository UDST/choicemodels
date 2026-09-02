"""
These are tests for the refactored choicemodels MNL codebase.

The estimation and prediction tests compare against fixed reference results generated
with `urbansim.urbanchoice.mnl` (UrbanSim 3.2) from the same seeded inputs, so they do
not require UrbanSim to be installed.

"""

import numpy as np
import pandas as pd
import pytest

from choicemodels import MultinomialLogit
from choicemodels.tools import MergedChoiceTable


MODEL_EXPRESSION = 'obsval + altval - 1'


@pytest.fixture
def data():
    """
    Seeded choosers and alternatives. The reference results below were generated from
    these exact tables, so the draw order must not change.

    """
    rng = np.random.default_rng(12345)
    d1 = {'oid': np.arange(100),
          'obsval': rng.random(100),
          'choice': rng.choice(np.arange(5), size=100)}
    d2 = {'aid': np.arange(5),
          'altval': rng.random(5)}
    return pd.DataFrame(d1).set_index('oid'), pd.DataFrame(d2).set_index('aid')

@pytest.fixture
def obs(data):
    return data[0]

@pytest.fixture
def alts(data):
    return data[1]


def test_mnl(obs, alts):
    """
    Confirm that MNL estimation runs, using the native estimator.

    """
    mct = MergedChoiceTable(obs, alts, 'choice')
    m = MultinomialLogit(mct, MODEL_EXPRESSION)
    print(m.fit())


def test_mnl_estimation(obs, alts):
    """
    Confirm that estimated params match reference results from urbansim.urbanchoice.

    """
    mct = MergedChoiceTable(obs, alts, 'choice')
    r = MultinomialLogit(mct, MODEL_EXPRESSION).fit().get_raw_results()

    assert r['log_likelihood']['null'] == pytest.approx(-160.94379124341)
    assert r['log_likelihood']['convergence'] == pytest.approx(-160.23827438299497)
    assert r['log_likelihood']['ratio'] == pytest.approx(0.0043836227229667735)

    fit = r['fit_parameters']
    # 'obsval' does not vary across alternatives, so its coefficient is identified at zero
    assert fit['Coefficient'].iloc[0] == pytest.approx(0.0, abs=1e-10)
    assert fit['Coefficient'].iloc[1] == pytest.approx(0.49098178188816505)
    np.testing.assert_allclose(
        fit['Std. Error'].values, [0.349133923117027, 0.2912928880390445], rtol=1e-6)
    assert fit['T-Score'].iloc[1] == pytest.approx(1.685526156142729)


def test_mnl_prediction(obs, alts):
    """
    Confirm that fitted probabilities match reference results from urbansim.urbanchoice.
    The alternative sampling is seeded so the choice table is reproducible.

    """
    np.random.seed(0)
    mct = MergedChoiceTable(obs, alts, 'choice', 5)
    results = MultinomialLogit(mct, model_expression=MODEL_EXPRESSION).fit()

    assert results.fitted_parameters[0] == pytest.approx(0.0, abs=1e-10)
    assert results.fitted_parameters[1] == pytest.approx(0.6623212935691077)

    probs = results.probabilities(mct)
    assert len(probs) == 500

    # probabilities sum to one within each choice scenario
    totals = probs.groupby(level=0).sum()
    np.testing.assert_allclose(totals.values, 1.0)

    expected_head = pd.Series(
        [0.2295914413632789, 0.18027237242448077, 0.18027237242448077,
         0.18027237242448077, 0.2295914413632789, 0.1986877750239994,
         0.15600719418213743, 0.1986877750239994, 0.2479294807458644,
         0.1986877750239994],
        index=pd.MultiIndex.from_tuples(
            [(99, 4), (99, 3), (99, 3), (99, 3), (99, 4),
             (98, 4), (98, 3), (98, 4), (98, 1), (98, 4)],
            names=['oid', 'aid']),
        name='prob')
    pd.testing.assert_series_equal(probs.iloc[:10], expected_head, check_index_type=False)
    assert (probs ** 2).sum() == pytest.approx(20.420676103008006)
