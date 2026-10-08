"""PF priorities only: the environment retains ownership of RB allocation.

This is an EdgeRIC-inspired simulator adapter, not an RT-E2 client or a
distributed RIC. Rates and historical averages are expressed in Mbps.
"""

import numpy as np


class EdgeRICScheduler:
    """Compute local PF weights and aggregate by the nearest covered UAV.

    epsilon_mbps is a rate-scale regularizer, not machine epsilon. The
    environment uses 5% of its maximum single-RB rate, bounding cold-start
    weights by 20 under its channel model. Thus zero history cannot create
    arbitrarily large priorities. This regularization biases pure PF near
    zero and should be included in later sensitivity experiments.

    Equal global weights are ordered by GT index (stable sort). Equal UAV
    distances select the smallest UAV index. Uncovered users have weight 0
    and selected_uav -1; the original allocator still enforces coverage.
    """

    def __init__(self, bandwidth_hz, tx_power_w, noise_psd_w_hz, epsilon_mbps):
        parameters = np.asarray(
            [bandwidth_hz, tx_power_w, noise_psd_w_hz, epsilon_mbps],
            dtype=np.float64,
        )
        if not np.isfinite(parameters).all() or (parameters <= 0).any():
            raise ValueError("Radio parameters and epsilon_mbps must be finite and positive")
        self.bandwidth_hz, self.tx_power_w, self.noise_psd_w_hz, self.epsilon_mbps = parameters
        self.reset()

    def reset(self):
        """Discard diagnostics from the preceding episode."""
        self.local_rates_mbps = None
        self.local_weights = None
        self.weights = None
        self.estimated_rates_mbps = None
        self.history_mbps = None
        self.selected_uav = None
        self.coverage = None
        self.priority = None

    def compute_weights(self, channel_gains, distances, coverage, avg_rate_mbps):
        """Use current pre-allocation gains and previous completed history.

        Inputs have shape (UAV, GT), except history with shape (GT,).
        No actual current SINR/rate or future allocation enters this method.
        Invalid/nonfinite telemetry is rejected instead of silently ordered.
        """
        gains = np.asarray(channel_gains, dtype=np.float64)
        distances = np.asarray(distances, dtype=np.float64)
        covered = np.asarray(coverage, dtype=bool)
        history = np.asarray(avg_rate_mbps, dtype=np.float64)
        if gains.ndim != 2 or gains.shape[0] == 0:
            raise ValueError("channel_gains must have shape (positive number of UAVs, GTs)")
        if distances.shape != gains.shape or covered.shape != gains.shape:
            raise ValueError("distances and coverage must match channel_gains")
        if history.shape != (gains.shape[1],):
            raise ValueError("history must have one entry per GT")
        for name, value in (("gains", gains), ("distances", distances), ("history", history)):
            if not np.isfinite(value).all() or (value < 0).any():
                raise ValueError(name + " must be finite and nonnegative")

        # log(1 + SNR) in the log domain also handles zero gains without
        # overflowing the intermediate SNR for very strong links.
        with np.errstate(divide="ignore"):
            log_snr = (np.log(self.tx_power_w) + np.log(gains)
                       - np.log(self.noise_psd_w_hz) - np.log(self.bandwidth_hz))
        rates = (self.bandwidth_hz * 1e-6 / np.log(2.0)) * np.logaddexp(0.0, log_snr)
        rates = np.where(covered, rates, 0.0)
        with np.errstate(over="ignore", invalid="ignore"):
            local_weights = rates / (history[None, :] + self.epsilon_mbps)
        if not np.isfinite(rates).all() or not np.isfinite(local_weights).all():
            raise ValueError("Radio parameters produce nonfinite rates or weights")

        nearest = np.argmin(np.where(covered, distances, np.inf), axis=0)
        any_coverage = covered.any(axis=0)
        gt_ids = np.arange(gains.shape[1])
        weights = np.where(any_coverage, local_weights[nearest, gt_ids], 0.0)

        self.coverage = covered.copy()
        self.history_mbps = history.copy()
        self.local_rates_mbps = rates
        self.local_weights = local_weights
        self.selected_uav = np.where(any_coverage, nearest, -1)
        self.estimated_rates_mbps = np.where(any_coverage, rates[nearest, gt_ids], 0.0)
        self.weights = weights
        self.priority = np.argsort(-weights, kind="stable")
        return weights.copy()

    def get_priority(self):
        """Return the current GT ordering without exposing mutable diagnostics."""
        if self.priority is None:
            raise RuntimeError("compute_weights must be called before get_priority")
        return self.priority.copy()
