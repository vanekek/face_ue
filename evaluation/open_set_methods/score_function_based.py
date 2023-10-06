import numpy as np
from .base_method import OpenSetMethod
from evaluation.open_set_methods.posterior_prob_based import PosteriorProb


class SimilarityBasedPrediction(OpenSetMethod):
    def __init__(
        self,
        kappa: float,
        beta: float,
        acceptance_score,
        uncertainty_function,
        alpha: float,
        T: float,
    ) -> None:
        super().__init__()
        self.kappa = kappa
        self.beta = beta
        self.acceptance_score = acceptance_score
        self.uncertainty_function = uncertainty_function
        self.alpha = alpha
        self.T = T

    def setup(self, similarity_matrix: np.ndarray):
        self.similarity_matrix = np.mean(similarity_matrix, axis=1)
        self.probe_score = self.acceptance_score(self.similarity_matrix)
        K = self.similarity_matrix.shape[-1]
        mises_maxprob = PosteriorProb(
            kappa=self.kappa, beta=self.beta, class_model="vMF", K=K
        )
        self.tau = (
            1
            / self.kappa
            * (
                np.log(self.beta / (1 - self.beta))
                + np.log(K)
                + mises_maxprob.log_uniform_dencity
                - mises_maxprob.log_normalizer
            )
        )

    def predict(self):
        predict_id = np.argmax(self.similarity_matrix, axis=-1)
        return predict_id, self.probe_score < self.tau

    def predict_uncertainty(self, data_uncertainty: np.ndarray):
        if data_uncertainty.shape[1] == 1:
            # here self.data_uncertainty is scf concetration
            data_uncertainty = -data_uncertainty[:, 0]
        else:
            raise NotImplemented
        unc = self.uncertainty_function(self.similarity_matrix, self.probe_score)
        unc_norm = (unc - np.min(unc)) / (np.max(unc) - np.min(unc))

        data_uncertainty_norm = (data_uncertainty - np.min(data_uncertainty)) / (
            np.max(data_uncertainty) - np.min(data_uncertainty)
        )
        data_conf_norm = (-data_uncertainty_norm + 1) ** (1 / self.T)
        conf_norm = -unc_norm + 1
        comb_conf = conf_norm * (1 - self.alpha) + data_conf_norm * self.alpha
        return -comb_conf
