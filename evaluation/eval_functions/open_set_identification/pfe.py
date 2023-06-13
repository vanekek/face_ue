from pathlib import Path

import numexpr as ne
import numpy as np

from ..metrics import compute_detection_and_identification_rate
from .abc import Abstract1NEval
from ..confidence_functions import AbstractConfidence


from evaluation.eval_functions.distaince_functions import compute_pfe_sim


class PFE(Abstract1NEval):
    def __init__(
        self, confidence_function: AbstractConfidence, variance_scale: float
    ) -> None:
        """
        Implements PFE “likelihood” of distributions belonging to the same person (sharing the same latent code)

        https://ieeexplore.ieee.org/document/9008376
        Eq. (3)
        """
        self.confidence_function = confidence_function
        self.variance_scale = variance_scale

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
        similarity = np.dot(probe_feats, gallery_feats.T)  # (19593, 1772)

        # compute pfe likelihood
        probe_feats = probe_feats
        probe_sigma_sq = probe_unc * self.variance_scale

        gallery_feats = gallery_feats
        gallery_sigma_sq = gallery_unc * self.variance_scale

        pfe_cache_path = Path("/app/cache/pfe_cache") / (
            "default_pfe_variance_shift_"
            + str(self.variance_scale)
            + f"_gallery_size_{gallery_feats.shape[1]}"
            + ".npy"
        )

        if pfe_cache_path.is_file():
            pfe_similarity = np.load(pfe_cache_path)
        else:
            pfe_similarity = compute_pfe_sim(
                probe_feats,
                gallery_feats,
                probe_sigma_sq,
                gallery_sigma_sq,
                pfe_cache_path=pfe_cache_path,
            )

        # compute confidences
        probe_score = self.confidence_function(pfe_similarity)

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
