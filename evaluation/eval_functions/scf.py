import confidence_functions
import numpy as np
import scipy
from metrics import compute_detection_and_identification_rate


class SCF:
    def __init__(
        self, confidence_function: dict, k_shift: float, use_cosine_sim_match: bool
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
        gallery_unc = gallery_unc[np.newaxis, :, 0]

        gallery_unc = gallery_unc + self.k_shift
        probe_unc = probe_unc + self.k_shift

        d = probe_feats.shape[1]
        k_i_times_k_j = probe_unc * gallery_unc
        mu_ij = 2 * np.dot(probe_feats, gallery_feats.T)
        k_ij = np.sqrt(probe_unc**2 + gallery_unc**2 + mu_ij * k_i_times_k_j)

        log_iv_i = (
            np.log(
                1e-6 + scipy.special.ive(d / 2 - 1, probe_unc, dtype=probe_unc.dtype)
            )
            + probe_unc
        )
        log_iv_j = (
            np.log(
                1e-6
                + scipy.special.ive(d / 2 - 1, gallery_unc, dtype=gallery_unc.dtype)
            )
            + gallery_unc
        )
        log_iv_ij = (
            np.log(1e-6 + scipy.special.ive(d / 2 - 1, k_ij, dtype=k_ij.dtype)) + k_ij
        )

        scf_similarity = (
            (d / 2 - 1) * (np.log(probe_unc) + np.log(gallery_unc) - np.log(k_ij))
            - (log_iv_i + log_iv_j - log_iv_ij)
            - d / 2 * np.log(2 * np.pi)
            - d * np.log(64)
        )
        # scf_similarity = -scf_similarity
        if self.use_cosine_sim_match:
            similarity = mu_ij / 2
        else:
            similarity = scf_similarity

        # compute confidences
        confidence_function = getattr(
            confidence_functions, self.confidence_function.class_name
        )(**self.confidence_function.init_args)
        probe_score = confidence_function(scf_similarity)

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
