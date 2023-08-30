import numpy as np
from .base_method import OpenSetMethod
from evaluation.confidence_functions import MisesProb


class vMFSumUnc(OpenSetMethod):
    def __init__(
        self, kappa: float, beta: float, uncertainty_type: str, alpha: float
    ) -> None:
        super().__init__()
        self.kappa = kappa
        self.beta = beta
        self.uncertainty_type = uncertainty_type
        self.alpha = alpha
        self.mises_maxprob = MisesProb(kappa=self.kappa, beta=self.beta)
        self.all_classes_log_prob = None

    def setup(self, similarity_matrix: np.ndarray):
        self.similarity_matrix = similarity_matrix
        self.all_classes_log_prob = (
            self.mises_maxprob.compute_all_class_log_probabilities(
                self.similarity_matrix
            )
        )
        self.all_classes_log_prob = np.mean(self.all_classes_log_prob, axis=1)

    def predict(self):
        predict_id = np.argmax(self.all_classes_log_prob, axis=-1)
        return predict_id

    def predict_uncertainty(self, data_uncertainty: np.ndarray):
        if self.uncertainty_type == "maxprob":
            unc = -np.max(self.all_classes_log_prob, axis=-1)
        elif self.uncertainty_type == "entr":
            unc = -np.sum(
                np.exp(self.all_classes_log_prob) * self.all_classes_log_prob, axis=-1
            )
        else:
            raise ValueError

        # normalize and sum with data uncertainty
        unc_norm = (unc - np.min(unc)) / (np.max(unc) - np.min(unc))

        if data_uncertainty.shape[1] == 1:
            # here data_uncertainty is scf concetration
            data_uncertainty = -data_uncertainty[:, 0]
        else:
            raise NotImplemented
        data_uncertainty_norm = (data_uncertainty - np.min(data_uncertainty)) / (
            np.max(data_uncertainty) - np.min(data_uncertainty)
        )

        comb_unc = unc_norm * (1 - self.alpha) + data_uncertainty_norm * self.alpha
        return comb_unc
