"""Sharma Absolute Boundary Summation (ABS) Model -- Python reference implementation.

    eta    = N / (N + S),  N = n(|a| + |b|),  S = sum of n iid uniform integers on [a, b]
    v_real = v_ideal * (1 + alpha * (eta - 1))

See the paper for proofs: eta is always defined and positive when b != 0,
its spread is set mainly by n, and E[eta] > 1 (use normalize=True to remove the bias).
"""
import random


class SharmaABS:
    def __init__(self, min_val: int, max_val: int, complexity: int,
                 seed=None, normalize: bool = False):
        if complexity <= 0:
            raise ValueError("complexity n must be >= 1")
        if min_val > max_val:
            min_val, max_val = max_val, min_val
        if max_val == 0:
            raise ValueError("max_val must be non-zero (otherwise N + S can be 0)")
        self.min_val = min_val
        self.max_val = max_val
        self.complexity = complexity
        self.N = complexity * (abs(min_val) + abs(max_val))
        self.rng = random.Random(seed)
        self.normalize = normalize
        self._mean = self.exact_mean() if normalize else 1.0

    def pmf_sum(self):
        """Exact PMF of S as a list, for S = n*a ... n*b."""
        m = self.max_val - self.min_val + 1
        p = [1.0]
        for _ in range(self.complexity):
            q = [0.0] * (len(p) + m - 1)
            for i, pi in enumerate(p):
                if pi:
                    w = pi / m
                    for j in range(m):
                        q[i + j] += w
            p = q
        return p

    def exact_mean(self) -> float:
        """Exact E[eta] (finite sum over the PMF of S)."""
        lo = self.complexity * self.min_val
        return sum(p * self.N / (self.N + lo + k) for k, p in enumerate(self.pmf_sum()))

    def compute_eta(self) -> float:
        s = sum(self.rng.randint(self.min_val, self.max_val) for _ in range(self.complexity))
        return self.N / (self.N + s) / self._mean

    @staticmethod
    def apply_sharma_noise(ideal_val: float, eta: float, alpha: float) -> float:
        """alpha in [0, 1] keeps the sign of ideal_val."""
        return ideal_val * (1.0 - alpha + alpha * eta)

    def generate_noisy_dataset(self, ideal_data, alpha: float):
        return [self.apply_sharma_noise(v, self.compute_eta(), alpha) for v in ideal_data]
