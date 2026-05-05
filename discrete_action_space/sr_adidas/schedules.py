"""Temperature and epsilon schedules for SR-ADIDAS."""

import math


class TauSchedule:
    """Exponential entropy-temperature anneal driven by ADI convergence.

    Mirrors ADIDAS Algorithm 2 line 15: halve tau whenever the running
    ADI estimate drops below the threshold, down to tau_min.
    """

    def __init__(self, tau_init=100.0, tau_min=1e-3, decay_factor=0.5, threshold=1e-3):
        self.tau = float(tau_init)
        self.tau_min = float(tau_min)
        self.decay_factor = float(decay_factor)
        self.threshold = float(threshold)

    def step(self, adi_estimate):
        if float(adi_estimate) < self.threshold and self.tau > self.tau_min:
            self.tau = max(self.tau_min, self.tau * self.decay_factor)

    def value(self):
        return self.tau


class EpsilonSchedule:
    """Linear or exponential schedule for the exploration / robustness radius."""

    def __init__(self, start=1.0, end=0.05, decay_fraction=0.6, total_steps=10_000,
                 mode="linear"):
        self.start = float(start)
        self.end = float(end)
        self.decay_fraction = float(decay_fraction)
        self.total_steps = int(total_steps)
        self.mode = mode
        self._step = 0

    def step(self):
        self._step += 1

    def value(self):
        if self.total_steps <= 1:
            return self.end
        decay_steps = max(1, int(self.total_steps * self.decay_fraction))
        if self.mode == "linear":
            t = min(self._step, decay_steps) / decay_steps
            return float(self.start + t * (self.end - self.start))
        # exponential
        t = min(self._step, decay_steps) / decay_steps
        log_start = math.log(max(self.start, 1e-12))
        log_end = math.log(max(self.end, 1e-12))
        return float(math.exp(log_start + t * (log_end - log_start)))
