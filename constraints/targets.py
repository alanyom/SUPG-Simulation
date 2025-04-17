import numpy as np
from scipy.stats import norm
from supg.config import TARGET_RECALL, TARGET_PRECISION, CONFIDENCE_LEVEL

class ConstraintChecker:
    @staticmethod
    def recall_bounds(z1: float, z2: float) -> tuple[float, float]:
        # finding ci
        total = z1 + z2
        if total == 0:
            return (0.0, 0.0)
        p = z1 / total
        se = np.sqrt(p * (1 - p) / total)
        z = norm.ppf(1 - (1 - CONFIDENCE_LEVEL) / 2)
        return (max(0, p - z * se), min(1, p + z * se))

    def meets_targets(self, z1: float, z2: float, fp: float) -> bool:
        # checking with ci
        recall_lower, _ = self.recall_bounds(z1, z2)
        precision_lower = z1 / (z1 + fp) if (z1 + fp) > 0 else 0.0
        return (recall_lower >= TARGET_RECALL) and (precision_lower >= TARGET_PRECISION)
