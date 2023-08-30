import numpy as np
from .base_method import OpenSetMethod

class SimilarityBasedPrediction(OpenSetMethod):
    def __init__(self, tau: float, acceptance_score) -> None:
        super().__init__()
        self.tau = tau
        self.acceptance_score = acceptance_score

    def setup(self, similarity_matrix: np.ndarray):
        self.similarity_matrix = np.mean(similarity_matrix, axis=1)

    def predict(self):
        predict_id = np.argmax(self.similarity_matrix, axis=-1)
        probe_score = self.acceptance_score(self.similarity_matrix)
        predict_id[probe_score < self.tau] = self.similarity_matrix.shape[
            -1
        ]  # reject class
        return predict_id

    def predict_uncertainty(self, data_uncertainty: np.ndarray):
        if data_uncertainty.shape[1] == 1:
            # here self.data_uncertainty is scf concetration
            data_uncertainty = -self.data_uncertainty[:, 0]
        else:
            raise NotImplemented
        return data_uncertainty
