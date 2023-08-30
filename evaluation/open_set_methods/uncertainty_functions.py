import numpy as np

from typing import Any


class BernoulliVariance:
    def __call__(self, similarity: np.ndarray, probe_score: np.ndarray) -> Any:
        probe_score_norm = (probe_score + 1) / 2
        unc_score = probe_score_norm * (1 - probe_score_norm)
        return unc_score
