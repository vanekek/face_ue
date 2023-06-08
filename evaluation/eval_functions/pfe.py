from pathlib import Path

import numexpr as ne
import numpy as np
from tqdm import tqdm
from ..metrics import compute_detection_and_identification_rate
from .abc import Abstract1NEval
from ..confidence_functions import AbstractConfidence


def compute_pfe(
    pfe_similarity,
    chunck_slice,
    probe_feats,
    probe_sigma_sq,
    gallery_feats,
    gallery_sigma_sq,
):
    probe_sigma_sq_slice = probe_sigma_sq[:, :, chunck_slice]
    gallery_sigma_sq_slice = gallery_sigma_sq[:, :, chunck_slice]
    probe_feats_slice = probe_feats[:, :, chunck_slice]
    gallery_feats_slice = gallery_feats[:, :, chunck_slice]
    sigma_sq_sum = ne.evaluate("probe_sigma_sq_slice + gallery_sigma_sq_slice")
    slice = ne.evaluate(
        "(probe_feats_slice - gallery_feats_slice)**2 / sigma_sq_sum + log(sigma_sq_sum)"
    )
    slice_sum = ne.evaluate("sum(slice, axis=2)")
    ne.evaluate("slice_sum + pfe_similarity", out=pfe_similarity)


class PFE(Abstract1NEval):
    def __init__(self, confidence_function: AbstractConfidence, variance_scale: float) -> None:
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
        probe_feats = probe_feats[:, np.newaxis, :]
        probe_sigma_sq = probe_unc[:, np.newaxis, :] * self.variance_scale

        gallery_feats = gallery_feats[np.newaxis, :, :]
        gallery_sigma_sq = gallery_unc[np.newaxis, :, :] * self.variance_scale

        pfe_cache_path = Path("/app/cache/pfe_cache") / (
            "default_pfe_variance_shift_"
            + str(self.variance_scale)
            + f"_gallery_size_{gallery_feats.shape[1]}"
            + ".npy"
        )

        if pfe_cache_path.is_file():
            pfe_similarity = np.load(pfe_cache_path)
        else:
            pfe_similarity = np.zeros_like(similarity)

            chunck_size = 128
            for d in tqdm(range(probe_feats.shape[2] // chunck_size)):
                compute_pfe(
                    pfe_similarity,
                    slice(d * chunck_size, (d + 1) * chunck_size),
                    probe_feats,
                    probe_sigma_sq,
                    gallery_feats,
                    gallery_sigma_sq,
                )
            pfe_similarity = -0.5 * pfe_similarity
            np.save(pfe_cache_path, pfe_similarity)

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
