import random

class SharmaABS:
    """
    Sharma Absolute Boundary Summation (ABS) Model
    """
    def __init__(self, min_val: int, max_val: int, complexity: int):
        self.min_val = min_val
        self.max_val = max_val
        self.complexity = complexity

    def compute_eta(self) -> float:
        """
        Computes the dimensionless noise factor.
        """
        if self.complexity <= 0:
            return 1.0
            
        minimest = self.complexity * self.min_val
        miximet = self.complexity * self.max_val
        
        sum_random = sum(random.randint(self.min_val, self.max_val) for _ in range(self.complexity))
        
        numerator = abs(minimest) + abs(miximet)
        denominator = numerator + sum_random
        
        # Stability Guard: Default to 1.0 if denominator is 0
        if denominator == 0:
            return 1.0
            
        return numerator / denominator

    @staticmethod
    def apply_sharma_noise(ideal_val: float, eta: float, alpha: float) -> float:
        """
        Applies the Sharma noise to an ideal value.
        alpha: Influence coefficient (0.0 for perfectly deterministic to 1.0 for maximum turbulence)
        """
        return ideal_val * (1.0 - alpha + alpha * eta)
        
    def generate_noisy_dataset(self, ideal_data: list[float], alpha: float) -> list[float]:
        """
        Generates a noisy dataset for an input list of deterministic values.
        """
        return [self.apply_sharma_noise(val, self.compute_eta(), alpha) for val in ideal_data]
