import numpy as np
from .base_method import OpenSetMethod
from scipy.special import ive, hyp0f1, loggamma


class PosteriorProbability(OpenSetMethod):
    def __init__(
        self,
        kappa: float,
        beta: float,
        uncertainty_type: str,
        alpha: float,
        logunc: bool,
        class_model: str,
    ) -> None:
        super().__init__()
        self.kappa = kappa
        self.beta = beta
        self.uncertainty_type = uncertainty_type
        self.alpha = alpha
        self.mises_maxprob = PosteriorProb(
            kappa=self.kappa, beta=self.beta, class_model=class_model
        )
        self.all_classes_log_prob = None
        self.logunc = logunc

    def setup(self, similarity_matrix: np.ndarray):
        self.similarity_matrix = similarity_matrix
        self.all_classes_log_prob = (
            self.mises_maxprob.compute_all_class_log_probabilities(
                self.similarity_matrix
            )
        )
        self.all_classes_log_prob = np.mean(self.all_classes_log_prob, axis=1)

    def predict(self):
        predict_id = np.argmax(self.all_classes_log_prob[:, :-1], axis=-1)
        return predict_id, np.argmax(self.all_classes_log_prob, axis=-1) == (
            self.all_classes_log_prob.shape[-1] - 1
        )

    def predict_uncertainty(self, data_uncertainty: np.ndarray):
        if self.uncertainty_type == "maxprob":
            unc = -np.max(self.all_classes_log_prob, axis=-1)
        elif self.uncertainty_type == "entr":
            unc = -np.sum(
                np.exp(self.all_classes_log_prob) * self.all_classes_log_prob, axis=-1
            )
        else:
            raise ValueError

        if self.logunc:
            unc = np.log(unc - np.min(unc) + 1e-16)
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


class PosteriorProb:
    def __init__(self, kappa: float, beta: float, d=512, class_model="vMF") -> None:
        """
        Performes K+1 class classification, with K being number of gallery classed and
        K+1-th class is ood class.
        returns probabilities p(K+1|z) = \frac{p(z|c)p(c)}{p(z)} that test emb z belongs to reject class K+1
        where
        1. p(z|K+1) is uniform disribution on sphere
        2. p(z|c), c \leq K is von Mises-Fisher (vMF) distribution with koncentration k
        3. p(K+1) = beta and p(1)= ... =p(K) = (1 - beta)/K

        :param kappa: koncentration for von Mises-Fisher (vMF) distribution
        :param beta: prior probability of ood sample, p(K+1)
        """
        self.kappa = kappa
        self.beta = beta
        self.n = d / 2
        self.class_model = class_model
        self.log_uniform_dencity = (
            loggamma(self.n, dtype=np.float64) - np.log(2) - self.n * np.log(np.pi)
        )

        if self.class_model == "vMF":
            self.alpha = hyp0f1(self.n, self.kappa**2 / 4, dtype=np.float64)
            self.log_iv = (
                np.log(ive(self.n - 1, self.kappa, dtype=np.float64)) + self.kappa
            )
            self.log_normalizer = (
                (self.n - 1) * np.log(kappa) - self.n * np.log(2 * np.pi) - self.log_iv
            )
        elif self.class_model == "power":
            log_alpha = (
                loggamma(d / 2)
                + loggamma(d - 1 + 2 * self.kappa)
                - self.kappa * np.log(2)
                - loggamma(d - 1 + self.kappa)
                - loggamma(d / 2 + self.kappa)
            )
            self.alpha = np.exp(log_alpha)
            self.log_normalizer = (
                loggamma(d - 1 + self.kappa)
                + loggamma(d / 2 + self.kappa)
                + (self.kappa - 1) * np.log(2)
                - (d / 2) * np.log(np.pi)
                - loggamma(d - 1 + 2 * self.kappa)
            )
        else:
            raise ValueError

    def compute_log_z_prob(self, similarities: np.ndarray):
        K = similarities.shape[-1]

        if self.class_model == "vMF":
            logit_sum = (
                np.sum(np.exp(similarities * self.kappa), axis=-1) * (1 - self.beta) / K
            )
        elif self.class_model == "power":
            logit_sum = (
                np.sum((1 + similarities) ** self.kappa, axis=-1) * (1 - self.beta) / K
            )

        log_z_prob = self.log_normalizer + np.log(logit_sum + self.alpha * self.beta)
        return log_z_prob

    def compute_all_class_log_probabilities(self, similarities: np.ndarray):
        log_z_prob = self.compute_log_z_prob(similarities)
        log_beta = np.log(self.beta)
        uniform_log_prob = self.log_uniform_dencity + log_beta - log_z_prob

        # compute gallery classes log prob
        K = similarities.shape[-1]
        log_z_prob = self.compute_log_z_prob(similarities)
        if self.class_model == "vMF":
            pz_c = self.kappa * similarities
        elif self.class_model == "power":
            pz_c = (1 + similarities) ** self.kappa
        gallery_log_probs = (
            self.log_normalizer
            + pz_c
            + np.log((1 - self.beta) / K)
            - log_z_prob[..., np.newaxis]
        )
        return np.concatenate(
            [gallery_log_probs, uniform_log_prob[..., np.newaxis]], axis=-1
        )
