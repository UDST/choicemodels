from collections import OrderedDict
import pickle
import warnings

import numpy as np
import pandas as pd
import pytest

from choicemodels import MultinomialLogit, MultinomialLogitResults
from choicemodels.pylogit_compat import create_design_matrix, predict_mnl
from choicemodels.tools import MergedChoiceTable


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


def test_design_matrix_supports_alternative_specific_coefficients(mode_choice_data):
    specification = OrderedDict([
        ("intercept", [2, 3]),
        ("travel_time", "all_diff"),
    ])
    design, names = create_design_matrix(
        mode_choice_data, specification, "alternative")

    assert design.shape == (8, 5)
    assert names == [
        "intercept_2", "intercept_3",
        "travel_time_1", "travel_time_2", "travel_time_3"]
    # each alternative-specific column carries the variable only in that alternative's rows
    np.testing.assert_array_equal(design[:, 2], [8, 0, 0, 15, 0, 14, 0, 0])
    np.testing.assert_array_equal(design[:, 3], [0, 11, 0, 0, 0, 0, 6, 0])
    np.testing.assert_array_equal(design[:, 4], [0, 0, 18, 0, 9, 0, 0, 10])


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


def test_flexible_mnl_accepts_merged_choice_table():
    """
    MergedChoiceTable frames carry the observation and alternative ids as index levels
    rather than columns (issue #77).

    """
    rng = np.random.default_rng(7)
    obs = pd.DataFrame({
        "oid": np.arange(60),
        "obsval": rng.random(60),
        "choice": rng.choice([1, 2, 3], size=60)}).set_index("oid")
    alts = pd.DataFrame({
        "aid": [1, 2, 3],
        "altval": rng.random(3)}).set_index("aid")
    mct = MergedChoiceTable(obs, alts, "choice")
    specification = OrderedDict([
        ("altval", "all_same"),
        ("obsval", [2, 3]),
    ])

    results = MultinomialLogit(mct, specification).fit()
    assert results.get_raw_results().estimation_success

    probabilities = results.probabilities(mct)
    assert probabilities.index.equals(mct.to_frame().index)
    np.testing.assert_allclose(probabilities.groupby(level="oid").sum(), 1.0)

    # ids as columns of a plain DataFrame give the same probabilities
    frame = mct.to_frame().reset_index()
    np.testing.assert_allclose(
        results.probabilities(frame).to_numpy(), probabilities.to_numpy())


def simulated_mode_choice(observation_count, seed=42):
    """
    Deterministic three-alternative mode choice data generated from known coefficients.

    """
    rng = np.random.default_rng(seed)
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

    return pd.DataFrame({
        "observation": observation,
        "alternative": alternative,
        "chosen": chosen,
        "intercept": np.ones(observation_count * 3),
        "travel_time": travel_time,
        "cost": cost,
    })



# Reference statistics generated with PyLogit 1.0.1 (NumPy 1.24, Pandas 1.5, SciPy 1.13)
# from the 500-observation `simulated_mode_choice()` table, exported to CSV so that
# both packages fit exactly the same data.

PYLOGIT_REFERENCE_GROUPED = {
    "params": [
        0.1574726496, -0.4800621963, -0.0747807344, -0.1923122476, -0.3248653054],
    "standard_errors": [
        0.1008727531, 0.5372382877, 0.0126392726, 0.0654889383, 0.1227090744],
    "tvalues": [
        1.56110193, -0.8935740569, -5.9165378431, -2.9365607775, -2.6474432058],
    "pvalues": [
        0.1184997028, 0.3715498321, 3.2879e-09, 0.0033187378, 0.0081102982],
    "robust_std_errs": [
        0.1011282695, 0.5499106512, 0.0128023604, 0.0644365108, 0.124850233],
    "robust_t_stats": [
        1.5571575613, -0.8729821749, -5.841167722, -2.9845229861, -2.6020400397],
    "robust_p_vals": [
        0.1194330815, 0.3826727936, 5.1836e-09, 0.0028402086, 0.0092671032],
    "cov": [
        [0.0101753123, 0.0054398941, -0.0000236993, -0.0000818412, -0.0000788703],
        [0.0054398941, 0.2886249778, -0.0002392709, 0.0157725455, -0.0553345664],
        [-0.0000236993, -0.0002392709, 0.0001597512, 0.0000080869, 0.0001118537],
        [-0.0000818412, 0.0157725455, 0.0000080869, 0.004288801, 0.0002896909],
        [-0.0000788703, -0.0553345664, 0.0001118537, 0.0002896909, 0.0150575169],
    ],
    "robust_cov": [
        [0.0102269269, 0.0069788351, -0.0000378466, -0.000090117, -0.0005360072],
        [0.0069788351, 0.3024017243, -0.0002949882, 0.016165842, -0.0579966222],
        [-0.0000378466, -0.0002949882, 0.0001639004, -0.0000068696, 0.0001094404],
        [-0.000090117, 0.016165842, -0.0000068696, 0.0041520639, 0.0001621365],
        [-0.0005360072, -0.0579966222, 0.0001094404, 0.0001621365, 0.0155875807],
    ],
    "log_likelihood": -480.48961450822986,
    "null_log_likelihood": -549.3061443340555,
    "rho_squared": 0.12527900977560424,
    "rho_bar_squared": 0.11617661750933594,
    "aic": 970.9792290164597,
    "bic": 992.0522695085707,
}

PYLOGIT_REFERENCE_ALTERNATIVE_SPECIFIC = {
    "params": [
        0.0177143753, -0.5854695401, -0.0722375331, -0.0651613583, -0.0936824722,
        -0.2200331087],
    "standard_errors": [
        0.5147302278, 0.6196319041, 0.0185759868, 0.0193617605, 0.0265977345,
        0.0585597209],
    "tvalues": [
        0.0344148728, -0.9448666801, -3.8887588505, -3.3654666023, -3.5221974275,
        -3.7574138867],
    "pvalues": [
        0.9725463237, 0.3447269442, 0.0001007582, 0.0007641432, 0.0004279853,
        0.0001716784],
    "robust_std_errs": [
        0.5132619885, 0.5999542227, 0.0184479793, 0.0201256982, 0.024884139,
        0.0574243392],
    "robust_t_stats": [
        0.0345133201, -0.9758570204, -3.9157423196, -3.2377191395, -3.7647463837,
        -3.8317046673],
    "robust_p_vals": [
        0.9724678207, 0.3291353234, 0.0000901265, 0.0012048936, 0.0001667182,
        0.0001272584],
    "cov": [
        [0.2649472074, 0.129285936, 0.0061518154, -0.0067920814, -0.0001815955,
         -0.0006946867],
        [0.129285936, 0.3839436966, 0.0065650003, 0.0002563614, -0.0128073871,
         0.0004206745],
        [0.0061518154, 0.0065650003, 0.0003450673, 0.0000311834, 0.0000115344,
         0.0000199077],
        [-0.0067920814, 0.0002563614, 0.0000311834, 0.0003748778, 0.0000203692,
         0.0000504036],
        [-0.0001815955, -0.0128073871, 0.0000115344, 0.0000203692, 0.0007074395,
         0.0000161471],
        [-0.0006946867, 0.0004206745, 0.0000199077, 0.0000504036, 0.0000161471,
         0.0034292409],
    ],
    "robust_cov": [
        [0.2634378688, 0.1191366941, 0.0057327951, -0.0071318627, -0.0001673845,
         -0.0000879992],
        [0.1191366941, 0.3599450694, 0.0066120265, 0.0006793817, -0.0113725604,
         0.0020828527],
        [0.0057327951, 0.0066120265, 0.0003403279, 0.0000459275, 0.000005177,
         0.0000416429],
        [-0.0071318627, 0.0006793817, 0.0000459275, 0.0004050437, 0.0000182904,
         0.0000359375],
        [-0.0001673845, -0.0113725604, 0.000005177, 0.0000182904, 0.0006192204,
         -0.0000460268],
        [-0.0000879992, 0.0020828527, 0.0000416429, 0.0000359375, -0.0000460268,
         0.0032975547],
    ],
    "log_likelihood": -480.5652564810391,
    "null_log_likelihood": -549.3061443340555,
    "rho_squared": 0.1251413051939434,
    "rho_bar_squared": 0.11421843447442137,
    "aic": 973.1305129620782,
    "bic": 998.4181615526114,
}


def assert_matches_pylogit(results, expected):
    """
    Coefficients match PyLogit to about 1e-8 and everything derived from them
    accordingly; the log-likelihood and fit measures match to machine precision.

    """
    for attribute in ["params", "standard_errors", "tvalues", "pvalues",
                      "robust_std_errs", "robust_t_stats", "robust_p_vals",
                      "cov", "robust_cov"]:
        np.testing.assert_allclose(
            getattr(results, attribute), expected[attribute],
            rtol=1e-6, atol=1e-8, err_msg=attribute)
    for attribute in ["log_likelihood", "null_log_likelihood", "rho_squared",
                      "rho_bar_squared", "aic", "bic"]:
        assert getattr(results, attribute) == pytest.approx(
            expected[attribute], rel=1e-9), attribute
    assert results.estimation_success


def test_matches_pylogit_reference_results():
    data = simulated_mode_choice(500)
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

    assert results.ind_var_names == ["ASC 2", "ASC 3", "Time", "Cost ground", "Cost 3"]
    assert (results.nobs, results.df_model, results.df_resid) == (500, 5, 495)
    assert_matches_pylogit(results, PYLOGIT_REFERENCE_GROUPED)


def test_matches_pylogit_reference_results_with_alternative_specific_coefficients():
    data = simulated_mode_choice(500)
    specification = OrderedDict([
        ("intercept", [2, 3]),
        ("travel_time", "all_diff"),
        ("cost", "all_same"),
    ])
    labels = OrderedDict([
        ("intercept", ["ASC 2", "ASC 3"]),
        ("travel_time", ["Time 1", "Time 2", "Time 3"]),
        ("cost", "Cost"),
    ])

    results = MultinomialLogit(
        data,
        specification,
        model_labels=labels,
        observation_id_col="observation",
        alternative_id_col="alternative",
        choice_col="chosen").fit().get_raw_results()

    assert results.ind_var_names == [
        "ASC 2", "ASC 3", "Time 1", "Time 2", "Time 3", "Cost"]
    assert (results.nobs, results.df_model, results.df_resid) == (500, 6, 494)
    assert_matches_pylogit(results, PYLOGIT_REFERENCE_ALTERNATIVE_SPECIFIC)


def test_predict_reuses_estimation_design(mode_choice_data, specification):
    results = MultinomialLogit(
        mode_choice_data,
        specification,
        observation_id_col="observation",
        alternative_id_col="alternative",
        choice_col="chosen").fit().get_raw_results()

    # no data: probabilities from the estimation design and fitted coefficients
    np.testing.assert_array_equal(results.predict(), results.long_fitted_probs)
    np.testing.assert_allclose(results.predict(), results.predict(mode_choice_data))

    # alternative coefficients, as in a simulation that varies the parameters
    uniform = results.predict(coefficients=np.zeros(5))
    np.testing.assert_allclose(uniform, [1 / 3, 1 / 3, 1 / 3, 1 / 2, 1 / 2, 1 / 3, 1 / 3, 1 / 3])
    np.testing.assert_allclose(
        results.predict(mode_choice_data, coefficients=np.zeros(5)), uniform)

    with pytest.raises(ValueError, match="one value per design column"):
        results.predict(coefficients=np.zeros(4))


def test_predict_rejects_mismatched_alternatives(mode_choice_data):
    """
    With alternative-specific coefficients the design columns come from the
    alternatives present in the data, so a table with a different set of
    alternatives cannot be scored with the fitted coefficients.

    """
    specification = OrderedDict([
        ("travel_time", "all_diff"),
    ])
    results = MultinomialLogit(
        mode_choice_data,
        specification,
        observation_id_col="observation",
        alternative_id_col="alternative",
        choice_col="chosen").fit().get_raw_results()

    relabeled = mode_choice_data.replace({"alternative": {3: 4}})
    with pytest.raises(ValueError, match="does not match the fitted model"):
        results.predict(relabeled)


@pytest.mark.parametrize("scale", [1e3, 1e6])
def test_prediction_survives_extreme_utilities(mode_choice_data, specification, scale):
    """
    Utilities far apart within a choice set would overflow or underflow a naive
    softmax. The probabilities must stay finite, strictly positive, and sum to
    one, so that log-probabilities computed downstream are finite too.

    """
    coefficients = np.array([scale, -scale, -scale, scale, -scale])

    probabilities = predict_mnl(
        mode_choice_data, "observation", "alternative", specification, coefficients)

    assert np.all(np.isfinite(probabilities))
    assert np.all(probabilities > 0)
    assert np.all(np.isfinite(np.log(probabilities)))
    totals = pd.Series(probabilities).groupby(mode_choice_data["observation"]).sum()
    np.testing.assert_allclose(totals, 1.0)
    # the dominant alternative in each choice set takes essentially all the mass
    assert np.isclose(pd.Series(probabilities).groupby(
        mode_choice_data["observation"]).max(), 1.0).all()

    results = MultinomialLogit(
        mode_choice_data,
        specification,
        observation_id_col="observation",
        alternative_id_col="alternative",
        choice_col="chosen").fit().get_raw_results()
    np.testing.assert_array_equal(results.predict(coefficients=coefficients), probabilities)


def test_converged_fit_reports_success():
    """
    Judged by the optimizer's own status, this fit stops with a "precision loss" message
    at the same coefficients as a successful fit; convergence is judged by the final
    gradient instead, so no non-convergence warning should be raised.

    """
    data = simulated_mode_choice(300)
    specification = OrderedDict([
        ("intercept", [2, 3]),
        ("travel_time", "all_same"),
        ("cost", [[1, 2], 3]),
    ])

    def fit(initial_coefs):
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            return MultinomialLogit(
                data,
                specification,
                observation_id_col="observation",
                alternative_id_col="alternative",
                choice_col="chosen",
                initial_coefs=initial_coefs).fit().get_raw_results()

    from_zero = fit(None)
    from_offset = fit(0.1)
    assert from_zero.estimation_success
    assert from_offset.estimation_success
    np.testing.assert_allclose(from_offset.params, from_zero.params, rtol=1e-6)
