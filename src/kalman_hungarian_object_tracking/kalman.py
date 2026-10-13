"""Constant-velocity Kalman filter with solve and Joseph covariance update."""

from dataclasses import dataclass

import numpy as np

from .config import TrackerConfig


def model_matrices(dt: float, sigma_a: float, sigma_det: float):
    if not np.isfinite([dt, sigma_a, sigma_det]).all() or dt <= 0 or sigma_a < 0 or sigma_det <= 0:
        raise ValueError("invalid motion/noise parameters")
    transition = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]], float)
    observation = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], float)
    acceleration = np.array([[dt**2 / 2, 0], [0, dt**2 / 2], [dt, 0], [0, dt]])
    process_noise = sigma_a**2 * (acceleration @ acceleration.T)
    observation_noise = sigma_det**2 * np.eye(2)
    return transition, observation, process_noise, observation_noise


@dataclass
class KalmanFilter:
    state: np.ndarray
    covariance: np.ndarray
    F: np.ndarray
    H: np.ndarray
    Q: np.ndarray
    R: np.ndarray

    @classmethod
    def initialize(cls, observation: np.ndarray, cfg: TrackerConfig):
        point = np.asarray(observation, dtype=float)
        if point.shape != (2,) or not np.isfinite(point).all():
            raise ValueError("initial observation must be a finite 2D point")
        result = cls(
            np.r_[point, [0.0, 0.0]],
            np.diag(cfg.initial_covariance_diag),
            *model_matrices(cfg.dt, cfg.sigma_a, cfg.sigma_det),
        )
        result._validate_state()
        return result

    def _validate_state(self):
        if self.state.shape != (4,) or self.covariance.shape != (4, 4):
            raise ValueError("invalid state or covariance shape")
        self.covariance = (self.covariance + self.covariance.T) / 2
        if not np.isfinite(self.state).all() or not np.isfinite(self.covariance).all():
            raise FloatingPointError("nonfinite Kalman state or covariance")
        tolerance = 1e-10 * max(1.0, float(np.linalg.norm(self.covariance, ord=2)))
        if np.linalg.eigvalsh(self.covariance).min() < -tolerance:
            raise FloatingPointError("Kalman covariance is not positive semidefinite")

    def predict(self) -> None:
        self.state = self.F @ self.state
        self.covariance = self.F @ self.covariance @ self.F.T + self.Q
        self._validate_state()

    @property
    def predicted_observation(self) -> np.ndarray:
        return self.H @ self.state

    @property
    def innovation_covariance(self) -> np.ndarray:
        return self.H @ self.covariance @ self.H.T + self.R

    def update(self, observation: np.ndarray) -> None:
        point = np.asarray(observation, dtype=float)
        if point.shape != (2,) or not np.isfinite(point).all():
            raise ValueError("observation must be a finite 2D point")
        residual = point - self.predicted_observation
        covariance = self.innovation_covariance
        gain = np.linalg.solve(covariance, self.H @ self.covariance).T
        self.state = self.state + gain @ residual
        remainder = np.eye(4) - gain @ self.H
        self.covariance = remainder @ self.covariance @ remainder.T + gain @ self.R @ gain.T
        self._validate_state()
