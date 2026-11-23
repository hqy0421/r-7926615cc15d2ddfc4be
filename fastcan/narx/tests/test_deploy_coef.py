"""Test NARX coefficient deployment and comparison on a fixed structure."""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from fastcan.narx import NARX, narx_prediction_diff, print_narx


def _make_data(rng, n_samples=300, max_delay=3, multi_output=False, nan=False):
    u0 = rng.uniform(0, 1, n_samples + max_delay)
    u1 = rng.normal(0, 0.1, n_samples)
    if multi_output:
        y0 = np.zeros(n_samples + max_delay)
        y1 = np.zeros(n_samples + max_delay)
        for i in range(max_delay, n_samples + max_delay):
            y0[i] = 0.5 * y0[i - 1] + 0.8 * y1[i - 1] + 0.3 * u0[i] ** 2 + 1
            y1[i] = 0.6 * y1[i - 1] + 0.2 * u0[i - 1] + 0.5
        y = np.c_[y0[max_delay:], y1[max_delay:]]
    else:
        y = np.zeros(n_samples + max_delay)
        for i in range(max_delay, n_samples + max_delay):
            y[i] = 0.5 * y[i - 1] + 0.3 * u0[i] ** 2 + 2 * u0[i - 1] * u0[i - 3] + 1
        y = y[max_delay:]
    e = rng.normal(0, 0.05, y.shape)
    y = y + e
    X = np.c_[u0[max_delay:], u1]
    if multi_output:
        feat_ids = np.array([[-1, 1], [0, 0], [0, 1]])
        delay_ids = np.array([[-1, 1], [0, 0], [1, 3]])
        output_ids = np.array([0, 0, 1])
    else:
        feat_ids = np.array([[-1, 1], [0, 0], [0, 1]])
        delay_ids = np.array([[-1, 1], [0, 0], [1, 3]])
        output_ids = None
    if nan:
        X = X.copy()
        y = y.copy()
        X[rng.choice(n_samples, 10, replace=False)] = np.nan
        y[rng.choice(n_samples, 10, replace=False)] = np.nan
    return X, y, feat_ids, delay_ids, output_ids


def test_fixed_coef_deploys_without_estimation():
    rng = np.random.default_rng(12345)
    X, y, feat_ids, delay_ids, output_ids = _make_data(rng)
    narx = NARX(feat_ids=feat_ids, delay_ids=delay_ids, output_ids=output_ids).fit(X, y)
    coef = np.r_[narx.coef_, narx.intercept_]

    deployed = NARX(feat_ids=feat_ids, delay_ids=delay_ids, output_ids=output_ids).fit(
        X, y, fixed_coef=coef
    )
    # Coefficients are assigned exactly, not re-estimated.
    assert_array_equal(deployed.coef_, narx.coef_)
    assert_array_equal(deployed.intercept_, narx.intercept_)

    # The structure is the same, and so is the prediction.
    assert_array_equal(deployed.feat_ids_, narx.feat_ids_)
    assert_array_equal(deployed.delay_ids_, narx.delay_ids_)
    assert_array_equal(deployed.output_ids_, narx.output_ids_)
    assert_allclose(deployed.predict(X), narx.predict(X))


def test_fixed_coef_is_not_affected_by_data_and_solver():
    rng = np.random.default_rng(12345)
    X, y, feat_ids, delay_ids, _ = _make_data(rng)
    narx = NARX(feat_ids=feat_ids, delay_ids=delay_ids).fit(X, y)
    coef = narx.get_coef()

    # Different data, sample weights, and solver must not change the
    # deployed coefficients, only the metadata inferred from the data.
    X_other, y_other, _, _, _ = _make_data(rng, n_samples=150)
    deployed = NARX(feat_ids=feat_ids, delay_ids=delay_ids).fit(
        X_other,
        y_other,
        sample_weight=np.ones(X_other.shape[0]),
        solver="minimize",
        fixed_coef=coef,
    )
    assert_array_equal(deployed.coef_, narx.coef_)
    assert_array_equal(deployed.intercept_, narx.intercept_)


def test_fixed_coef_ar_model():
    rng = np.random.default_rng(12345)
    rng2 = np.random.default_rng(0)
    n_samples = 200
    max_delay = 2
    y = np.zeros(n_samples + max_delay)
    for i in range(max_delay, n_samples + max_delay):
        y[i] = 0.5 * y[i - 1] + 0.3 * y[i - 2] + 1
    y = y[max_delay:] + rng2.normal(0, 0.01, n_samples)
    feat_ids = np.array([[0], [0]])
    delay_ids = np.array([[1], [2]])
    narx = NARX(feat_ids=feat_ids, delay_ids=delay_ids).fit(None, y)
    deployed = NARX(feat_ids=feat_ids, delay_ids=delay_ids).fit(
        None, y, fixed_coef=narx.get_coef()
    )
    assert_array_equal(deployed.coef_, narx.coef_)
    assert_array_equal(deployed.intercept_, narx.intercept_)


def test_fixed_coef_fit_intercept_false():
    rng = np.random.default_rng(12345)
    X = rng.random((50, 1))
    y = 2.0 * X[:, 0]
    feat_ids = np.array([[0]])
    delay_ids = np.array([[0]])
    narx = NARX(feat_ids=feat_ids, delay_ids=delay_ids, fit_intercept=False).fit(X, y)
    assert narx.get_coef().shape == (1,)
    deployed = NARX(feat_ids=feat_ids, delay_ids=delay_ids, fit_intercept=False).fit(
        X, y, fixed_coef=narx.get_coef()
    )
    assert_array_equal(deployed.coef_, narx.coef_)
    assert_array_equal(deployed.intercept_, [0.0])


def test_fixed_coef_rejects_conflicts():
    rng = np.random.default_rng(12345)
    X, y, feat_ids, delay_ids, _ = _make_data(rng)
    narx = NARX(feat_ids=feat_ids, delay_ids=delay_ids).fit(X, y)
    coef = narx.get_coef()
    n_terms = narx.coef_.size
    n_outputs = narx.n_outputs_

    with pytest.raises(ValueError, match=r"`fixed_coef` should have the shape of"):
        NARX(feat_ids=feat_ids, delay_ids=delay_ids).fit(X, y, fixed_coef=coef[:-1])
    with pytest.raises(ValueError, match=r"`fixed_coef` should have the shape of"):
        NARX(feat_ids=feat_ids, delay_ids=delay_ids).fit(
            X, y, fixed_coef=np.r_[coef, 0.0]
        )
    with pytest.raises(ValueError, match=r"should be a 1-dimensional array"):
        NARX(feat_ids=feat_ids, delay_ids=delay_ids).fit(
            X, y, fixed_coef=coef.reshape(n_terms + n_outputs, 1)
        )
    with pytest.raises(ValueError, match=r"cannot be used at the same time"):
        NARX(feat_ids=feat_ids, delay_ids=delay_ids).fit(
            X, y, coef_init=coef, fixed_coef=coef
        )


@pytest.mark.parametrize("multi_output", [False, True])
@pytest.mark.parametrize("nan", [False, True])
def test_reestimate_coefficients_on_new_batch(multi_output, nan):
    """Re-estimate the coefficients of a fixed structure on a new batch."""
    rng = np.random.default_rng(12345)
    X_old, y_old, feat_ids, delay_ids, output_ids = _make_data(
        rng, multi_output=multi_output, nan=nan
    )
    X_new, y_new, _, _, _ = _make_data(rng, multi_output=multi_output, nan=nan)

    # The maintenance team fixes the structure once.
    old_model = NARX(feat_ids=feat_ids, delay_ids=delay_ids, output_ids=output_ids).fit(
        X_old, y_old
    )

    # Only the coefficients are re-estimated on the new batch.
    new_model = NARX(feat_ids=feat_ids, delay_ids=delay_ids, output_ids=output_ids).fit(
        X_new, y_new
    )

    # The term order is not re-selected.
    assert_array_equal(new_model.feat_ids_, old_model.feat_ids_)
    assert_array_equal(new_model.delay_ids_, old_model.delay_ids_)
    assert_array_equal(new_model.output_ids_, old_model.output_ids_)
    assert new_model.coef_.shape == old_model.coef_.shape
    assert np.any(new_model.coef_ != old_model.coef_)


@pytest.mark.parametrize("multi_output", [False, True])
def test_narx_prediction_diff_same_window(multi_output):
    """Old and new coefficients are compared on the same data window."""
    rng = np.random.default_rng(12345)
    X_eval, y_eval, feat_ids, delay_ids, output_ids = _make_data(
        rng, multi_output=multi_output
    )
    X_old, y_old, _, _, _ = _make_data(rng, multi_output=multi_output)
    X_new, y_new, _, _, _ = _make_data(rng, multi_output=multi_output)

    old_model = NARX(feat_ids=feat_ids, delay_ids=delay_ids, output_ids=output_ids).fit(
        X_old, y_old
    )
    new_model = NARX(feat_ids=feat_ids, delay_ids=delay_ids, output_ids=output_ids).fit(
        X_new, y_new
    )

    y_init = y_eval[: old_model.max_delay_]
    y_old_pred = old_model.predict(X_eval, y_init=y_init)
    y_new_pred = new_model.predict(X_eval, y_init=y_init)
    y_diff = narx_prediction_diff(old_model, new_model, X_eval, y_init=y_init)
    assert_allclose(y_diff, y_new_pred - y_old_pred)

    # Deploying the old coefficients on the new model must reproduce the
    # old prediction, so the difference is exactly zero.
    new_model_deployed = NARX(
        feat_ids=feat_ids, delay_ids=delay_ids, output_ids=output_ids
    ).fit(X_new, y_new, fixed_coef=old_model.get_coef())
    assert_allclose(
        narx_prediction_diff(old_model, new_model_deployed, X_eval, y_init=y_init),
        0.0,
    )

    # Swapping the arguments flips the sign.
    assert_allclose(
        y_diff, -narx_prediction_diff(new_model, old_model, X_eval, y_init=y_init)
    )


def test_narx_prediction_diff_ar_model():
    rng = np.random.default_rng(0)
    n_samples = 100
    max_delay = 2
    y = np.zeros(n_samples + max_delay)
    for i in range(max_delay, n_samples + max_delay):
        y[i] = 0.5 * y[i - 1] + 0.3 * y[i - 2] + 1
    y = y[max_delay:] + rng.normal(0, 0.01, n_samples)
    feat_ids = np.array([[0], [0]])
    delay_ids = np.array([[1], [2]])
    old_model = NARX(feat_ids=feat_ids, delay_ids=delay_ids).fit(None, y)
    new_model = NARX(feat_ids=feat_ids, delay_ids=delay_ids).fit(
        None, y, fixed_coef=np.r_[old_model.coef_, old_model.intercept_ + 0.5]
    )
    y_diff = narx_prediction_diff(old_model, new_model, 20)
    assert y_diff.shape == (20,)
    y_init = y[:max_delay]
    y_diff_init = narx_prediction_diff(old_model, new_model, 20, y_init=y_init)
    assert_allclose(y_diff, y_diff_init)
    # The first `max_delay_` steps are the initial values and are shared.
    assert np.all(y_diff[:max_delay] == 0)
    assert np.all(y_diff[max_delay:] != 0)


def test_narx_prediction_diff_rejects_mismatched_structure():
    rng = np.random.default_rng(12345)
    X, y, feat_ids, delay_ids, _ = _make_data(rng)
    model_a = NARX(feat_ids=feat_ids, delay_ids=delay_ids).fit(X, y)

    # Different term order.
    swapped_feat_ids = feat_ids[::-1]
    swapped_delay_ids = delay_ids[::-1]
    model_b = NARX(feat_ids=swapped_feat_ids, delay_ids=swapped_delay_ids).fit(X, y)
    with pytest.raises(ValueError, match=r"different term structures"):
        narx_prediction_diff(model_a, model_b, X)

    # Different number of terms.
    model_c = NARX(feat_ids=feat_ids[:2], delay_ids=delay_ids[:2]).fit(X, y)
    with pytest.raises(ValueError, match=r"different term structures"):
        narx_prediction_diff(model_a, model_c, X)

    # Unfitted model.
    with pytest.raises(Exception, match=r"not fitted"):
        narx_prediction_diff(NARX(feat_ids=feat_ids, delay_ids=delay_ids), model_a, X)


def test_fixed_coef_print_narx_shows_deployed_coef(capsys):
    rng = np.random.default_rng(12345)
    X, y, feat_ids, delay_ids, _ = _make_data(rng, n_samples=100)
    narx = NARX(feat_ids=feat_ids, delay_ids=delay_ids).fit(X, y)
    coef = narx.get_coef()
    coef[:-1] += 0.5
    deployed = NARX(feat_ids=feat_ids, delay_ids=delay_ids).fit(X, y, fixed_coef=coef)
    print_narx(deployed)
    out = capsys.readouterr().out
    # The printed coefficients are the deployed ones, in the fixed term order.
    for value in deployed.coef_:
        assert f"{value:.3f}" in out
    for value in deployed.intercept_:
        assert f"{value:.3f}" in out
    assert out.index("Intercept") < out.index("X[k-1")
