from dataclasses import replace

import numpy as np
import pytest
from scipy.stats import chi2

from kalman_hungarian_object_tracking.association import kf_costs
from kalman_hungarian_object_tracking.kalman import KalmanFilter, model_matrices


def test_model_matrices_and_constant_velocity(config):
    F, H, Q, R = model_matrices(2, 0.5, 3)
    G = np.array([[2, 0], [0, 2], [2, 0], [0, 2]], float)
    np.testing.assert_allclose(Q, 0.25 * G @ G.T)
    np.testing.assert_array_equal(R, np.eye(2) * 9)
    np.testing.assert_array_equal(H @ np.array([1, 2, 3, 4]), [1, 2])
    cfg = replace(config.tracker, dt=2, sigma_a=0.5, sigma_det=3)
    kf = KalmanFilter.initialize(np.array([1, 2]), cfg)
    kf.state[2:] = [3, 4]
    initial_covariance = kf.covariance.copy()
    kf.predict()
    np.testing.assert_array_equal(kf.state, [7, 10, 3, 4])
    np.testing.assert_allclose(kf.covariance, F @ initial_covariance @ F.T + Q)


def test_analytic_update_and_uncertainty_reduction(config):
    kf = KalmanFilter.initialize(np.zeros(2), config.tracker)
    kf.update(np.array([1, 2]))
    np.testing.assert_allclose(kf.state, [0.8, 1.6, 0, 0], atol=1e-14)
    np.testing.assert_allclose(kf.covariance, np.diag([0.8, 0.8, 4, 4]), atol=1e-14)


def test_update_with_cross_covariance_has_correct_gain(config):
    kf = KalmanFilter.initialize(np.zeros(2), config.tracker)
    kf.predict()
    prior = kf.covariance.copy()
    # Independent scalar calculation: Pxx=8+sigma_a²/4, Pvx=4+sigma_a²/2.
    s = 9 + config.tracker.sigma_a**2 / 4
    kf.update(np.array([1, -2]))
    np.testing.assert_allclose(
        kf.state, [prior[0, 0] / s, -2 * prior[1, 1] / s, prior[2, 0] / s, -2 * prior[3, 1] / s]
    )
    assert np.trace(kf.covariance) < np.trace(prior)


@pytest.mark.parametrize("noise", [1e-4, 1.0, 100.0])
def test_joseph_covariance_psd_finite_and_symmetric_after_misses(config, noise):
    kf = KalmanFilter.initialize(np.zeros(2), replace(config.tracker, sigma_det=noise))
    rng = np.random.default_rng(3)
    for frame in range(200):
        kf.predict()
        if frame < 50 or frame > 150:
            kf.update(np.array([frame * 0.5, frame * 0.1]) + rng.normal(0, noise, 2))
        assert np.isfinite(kf.state).all() and np.isfinite(kf.covariance).all()
        np.testing.assert_allclose(kf.covariance, kf.covariance.T, atol=1e-12)
        assert np.linalg.eigvalsh(kf.covariance).min() >= -1e-10


def test_mahalanobis_cost_and_chi_square_gate(config):
    costs = kf_costs(
        np.array([[0.0, 0.0]]),
        np.array([[[4.0, 0.0], [0.0, 1.0]]]),
        np.array([[2.0, 2.0], [4.0, 0.0]]),
    )
    np.testing.assert_allclose(costs, [[5.0, 4.0]])
    assert config.tracker.kf_gate_mahalanobis_sq == pytest.approx(chi2.ppf(0.99, 2), abs=1e-6)


def test_filter_rejects_nonfinite_observations(config):
    with pytest.raises(ValueError):
        KalmanFilter.initialize(np.array([np.inf, 0]), config.tracker)
    kf = KalmanFilter.initialize(np.zeros(2), config.tracker)
    with pytest.raises(ValueError):
        kf.update(np.array([0.0, np.nan]))
