from collections import OrderedDict
import pickle

import numpy as np
import pandas as pd
import pytest

from choicemodels import MultinomialLogit, MultinomialLogitResults
from choicemodels.pylogit_compat import create_design_matrix


@pytest.fixture
def mode_choice_data():
    return pd.DataFrame({
        "observation": [1, 1, 1, 2, 2, 3, 3, 3],
        "alternative": [1, 2, 3, 1, 3, 1, 2, 3],
        "chosen": [1, 0, 0, 0, 1, 0, 1, 0],
        "travel_time": [8, 11, 18, 15, 9, 14, 6, 10],
        "cost": [5, 2, 1, 6, 2, 4, 3, 2],
        "intercept": np.ones(8),
    })


@pytest.fixture
def specification():
    return OrderedDict([
        ("intercept", [2, 3]),
        ("travel_time", "all_same"),
        ("cost", [[1, 2], 3]),
    ])


def test_design_matrix_supports_generic_alternative_and_grouped_coefficients(
        mode_choice_data, specification):
    design, names = create_design_matrix(
        mode_choice_data, specification, "alternative")

    assert design.shape == (8, 5)
    assert names == [
        "intercept_2", "intercept_3", "travel_time",
        "cost_[1, 2]", "cost_3"]
    np.testing.assert_array_equal(design[:, 0], [0, 1, 0, 0, 0, 0, 1, 0])
    np.testing.assert_array_equal(design[:, 1], [0, 0, 1, 0, 1, 0, 0, 1])


def test_names_are_validated_per_specification_entry(mode_choice_data):
    specification = OrderedDict([
        ("intercept", [2, 3]),
        ("travel_time", "all_same"),
    ])
    names = OrderedDict([
        ("intercept", ["ASC 2"]),
        ("travel_time", ["Time", "Extra"]),
    ])

    with pytest.raises(ValueError, match="names for 'intercept'"):
        create_design_matrix(mode_choice_data, specification, "alternative", names)


def test_flexible_mnl_fit_prediction_and_statistics(mode_choice_data, specification):
    model = MultinomialLogit(
        mode_choice_data,
        specification,
        observation_id_col="observation",
        alternative_id_col="alternative",
        choice_col="chosen")
    results = model.fit()
    raw = results.get_raw_results()

    assert model.estimation_engine == "PyLogit"
    assert len(results.fitted_parameters) == 5
    assert raw.params.index.tolist() == raw.ind_var_names
    assert raw.cov.shape == (5, 5)
    assert np.isfinite(raw.log_likelihood)
    assert raw.nobs == 3
    assert "Multinomial Logit" in results.report_fit()

    probabilities = results.probabilities(mode_choice_data)
    totals = probabilities.groupby(mode_choice_data["observation"]).sum()
    np.testing.assert_allclose(totals, 1.0)


def test_flexible_mnl_results_survive_pickle(mode_choice_data, specification, tmp_path):
    results = MultinomialLogit(
        mode_choice_data,
        specification,
        observation_id_col="observation",
        alternative_id_col="alternative",
        choice_col="chosen").fit()

    restored = pickle.loads(pickle.dumps(results))
    np.testing.assert_allclose(
        results.probabilities(mode_choice_data),
        restored.probabilities(mode_choice_data))

    model_path = tmp_path / "model.pkl"
    results.get_raw_results().to_pickle(model_path)
    with model_path.open("rb") as stream:
        restored_raw = pickle.load(stream)
    np.testing.assert_allclose(
        results.get_raw_results().predict(mode_choice_data),
        restored_raw.predict(mode_choice_data))


def test_flexible_mnl_results_restore_without_estimator(
        mode_choice_data, specification):
    fitted = MultinomialLogit(
        mode_choice_data,
        specification,
        observation_id_col="observation",
        alternative_id_col="alternative",
        choice_col="chosen").fit()
    restored = MultinomialLogitResults(
        model_expression=specification,
        model_labels=None,
        fitted_parameters=fitted.fitted_parameters,
        estimation_engine="PyLogit",
        observation_id_col="observation",
        alternative_id_col="alternative")

    assert restored.get_raw_results() is None
    np.testing.assert_allclose(
        fitted.probabilities(mode_choice_data),
        restored.probabilities(mode_choice_data))


def test_initial_values_match_expanded_specification(mode_choice_data, specification):
    model = MultinomialLogit(
        mode_choice_data,
        specification,
        observation_id_col="observation",
        alternative_id_col="alternative",
        choice_col="chosen",
        initial_coefs=[0, 0])

    with pytest.raises(ValueError, match="one value per coefficient"):
        model.fit()


def test_matches_pylogit_reference_results():
    rng = np.random.default_rng(42)
    observation_count = 500
    alternative = np.tile([1, 2, 3], observation_count)
    observation = np.repeat(np.arange(observation_count), 3)
    travel_time = rng.normal(20, 5, observation_count * 3)
    cost = rng.normal(4, 1, observation_count * 3)
    design = np.column_stack([
        alternative == 2,
        alternative == 3,
        travel_time,
        cost * (alternative != 3),
        cost * (alternative == 3),
    ])
    generating_coefs = np.array([0.2, -0.4, -0.08, -0.2, -0.35])
    utility = design.dot(generating_coefs).reshape(observation_count, 3)
    probability = np.exp(utility - utility.max(axis=1, keepdims=True))
    probability /= probability.sum(axis=1, keepdims=True)
    chosen = np.zeros(observation_count * 3, dtype=int)
    for obs in range(observation_count):
        chosen[obs * 3 + rng.choice(3, p=probability[obs])] = 1

    data = pd.DataFrame({
        "observation": observation,
        "alternative": alternative,
        "chosen": chosen,
        "intercept": np.ones(observation_count * 3),
        "travel_time": travel_time,
        "cost": cost,
    })
    specification = OrderedDict([
        ("intercept", [2, 3]),
        ("travel_time", "all_same"),
        ("cost", [[1, 2], 3]),
    ])
    labels = OrderedDict([
        ("intercept", ["ASC 2", "ASC 3"]),
        ("travel_time", "Time"),
        ("cost", ["Cost ground", "Cost 3"]),
    ])

    results = MultinomialLogit(
        data,
        specification,
        model_labels=labels,
        observation_id_col="observation",
        alternative_id_col="alternative",
        choice_col="chosen").fit().get_raw_results()

    # Generated with PyLogit 1.0.1 from the same deterministic input.
    expected = np.array([
        0.1574726496, -0.4800621963, -0.0747807344,
        -0.1923122476, -0.3248653054])
    np.testing.assert_allclose(results.params, expected, rtol=1e-7, atol=1e-8)
    assert results.log_likelihood == pytest.approx(-480.48961450823003)
    assert results.estimation_success
