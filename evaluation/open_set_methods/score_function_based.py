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
        T_data_unc: float,
        kappa_is_tau: bool,
    ) -> None:
        super().__init__()
        self.kappa = kappa
        self.beta = beta
        self.acceptance_score = acceptance_score
        self.uncertainty_function = uncertainty_function
        self.alpha = alpha
        self.T = T
        self.T_data_unc = T_data_unc
        self.kappa_is_tau = kappa_is_tau

    def setup(self, similarity_matrix: np.ndarray):
        self.similarity_matrix = np.mean(similarity_matrix, axis=1)
        self.probe_score = self.acceptance_score(self.similarity_matrix)
        K = self.similarity_matrix.shape[-1]
        if self.kappa_is_tau:
            self.tau = self.kappa
        else:
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
            data_uncertainty = data_uncertainty[:, 0]
        else:
            raise NotImplemented
        unc = self.uncertainty_function(
            self.similarity_matrix, self.probe_score, self.tau
        )
        unc_norm = (unc - np.min(unc)) / (np.max(unc) - np.min(unc))

        min_kappa = 150
        max_kappa = 2700
        data_uncertainty_norm = (data_uncertainty - min_kappa) / (max_kappa - min_kappa)
        assert np.sum(data_uncertainty_norm < 0) == 0
        # data_uncertainty_norm = data_uncertainty
        data_conf_norm = (data_uncertainty_norm) ** (1 / self.T_data_unc)

        conf_norm = (-unc_norm + 1) ** (1 / self.T)
        comb_conf = conf_norm * (1 - self.alpha) + data_conf_norm * self.alpha
        return -comb_conf
