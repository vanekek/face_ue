import numpy as np


class MeanEstimate:
    def __init__(self, K) -> None:
        self.K = K

    def __call__(self, mu_verif, sigma_verif):
        """
        sigma: num_verif x 512
        mu: num_verif x 512

        return:
        Z with shape n x num_z_samples x 512
        z_weights num_z_samples
        """
        return mu_verif[:, np.newaxis, :], np.ones(1)


class HighSigma:
    def __init__(self, K, s) -> None:
        self.K = K
        self.s = s

    def __call__(self, mu_verif, sigma_verif):
        """
        sigma: num_verif x 512
        mu: num_verif x 512

        return:
        Z with shape n x num_z_samples x 512
        """
        sigma_top_k_idx = np.flip(np.argsort(sigma_verif, axis=1), axis=1)[:, : self.K]
        z_plus = mu_verif[:, np.newaxis, :]  # + #self.s *
        z = np.moveaxis(z, 2, 1)  # n x 512 x num_z_samples
        a_ilj_final = mu @ z
        assert a_ilj_final.shape[0] == z.shape[0]
        a_ilj_final = np.moveaxis(a_ilj_final, 2, 1)  # n x num_z_samples x K
        return a_ilj_final / self.T
