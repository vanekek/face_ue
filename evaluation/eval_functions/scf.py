import numpy as np
from ..metrics import compute_detection_and_identification_rate
from .abc import Abstract1NEval
from ..confidence_functions import AbstractConfidence


def compute_scf_sim(
    mu_ij: np.ndarray,
    X_unc: np.ndarray,
    Y_unc: np.ndarray, 
    d: int
):
    from scipy.special import ive
    X_unc = X_unc[None, :, 0]
    k_i_times_k_j = Y_unc * X_unc
    k_ij = np.sqrt(Y_unc**2 + X_unc**2 + mu_ij * k_i_times_k_j)

    log_iv_i = (
        np.log(
            1e-6 + ive(d / 2 - 1, Y_unc, dtype=Y_unc.dtype)
        )
        + Y_unc
    )
    log_iv_j = (
        np.log(
            1e-6
            + ive(d / 2 - 1, X_unc, dtype=X_unc.dtype)
        )
        + X_unc
    )
    log_iv_ij = (
        np.log(1e-6 + ive(d / 2 - 1, k_ij, dtype=k_ij.dtype)) + k_ij
    )

    scf_similarity = (
        (d / 2 - 1) * (np.log(Y_unc) + np.log(X_unc) - np.log(k_ij)) # type: ignore
        - (log_iv_i + log_iv_j - log_iv_ij)
        - d / 2 * np.log(2 * np.pi)
        - d * np.log(64)
    )
    
    return scf_similarity


class SCF(Abstract1NEval):
    def __init__(
        self, confidence_function: AbstractConfidence, k_shift: float, use_cosine_sim_match: bool
    ) -> None:
        """
        Implements SCF mutual “likelihood” of distributions belonging to the same person (sharing the same latent code)

        https://ieeexplore.ieee.org/document/9577756
        Eq. (13)
        """
        self.confidence_function = confidence_function
        self.k_shift = k_shift
        self.use_cosine_sim_match = use_cosine_sim_match

    def __call__(
        self,
        probe_feats,
        probe_unc,
        gallery_feats,
        gallery_unc,
        probe_ids,
        gallery_ids,
        fars,
    ):
        print(
            "probe_feats: %s, gallery_feats: %s"
            % (probe_feats.shape, gallery_feats.shape)
        )
        
        gallery_unc = gallery_unc + self.k_shift
        probe_unc = probe_unc + self.k_shift

        mu_ij = 2 * np.dot(probe_feats, gallery_feats.T)
        d = probe_feats.shape[1]
        scf_similarity = compute_scf_sim(mu_ij, gallery_unc, probe_unc, d)
        # scf_similarity = -scf_similarity
        if self.use_cosine_sim_match:
            similarity = mu_ij / 2
        else:
            similarity = scf_similarity

        # compute confidences
        probe_score = self.confidence_function(scf_similarity)

        # Compute Detection & identification rate for open set recognition
        (
            top_1_count,
            top_5_count,
            top_10_count,
            threshes,
            recalls,
            cmc_scores,
        ) = compute_detection_and_identification_rate(
            fars, probe_ids, gallery_ids, similarity, probe_score
        )
        return top_1_count, top_5_count, top_10_count, threshes, recalls, cmc_scores
