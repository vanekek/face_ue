from pathlib import Path

import numpy as np

# from ..metrics import compute_detection_and_identification_rate
from evaluation.eval_functions.open_set_identification.abc import Abstract1NEval
from evaluation.confidence_functions import AbstractConfidence

from tqdm import tqdm
from evaluation.eval_functions.distaince_functions import compute_pfe


class PFE(Abstract1NEval):
    def __init__(
        self,
        confidence_function: AbstractConfidence,
        variance_scale: float,
        use_cosine_sim_match: bool,
    ) -> None:
        """
        Implements PFE “likelihood” of distributions belonging to the same person (sharing the same latent code)

        https://ieeexplore.ieee.org/document/9008376
        Eq. (3)
        """
        self.confidence_function = confidence_function
        self.variance_scale = variance_scale
        self.use_cosine_sim_match = use_cosine_sim_match
        # self.compute_pfe_sim = PfeSim()

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
            Path("/app/cache/pfe_cache").mkdir(exist_ok=True)
            pfe_similarity = np.zeros((probe_feats.shape[0], gallery_feats.shape[1]))

            chunck_size = 128
            assert probe_feats.shape[2] % chunck_size == 0
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

        if self.use_cosine_sim_match is False:
            similarity = pfe_similarity
        else:
            similarity = np.dot(probe_feats, gallery_feats.T)  # (19593, 1772)

        # compute confidences
        probe_score = self.confidence_function(pfe_similarity)
        return similarity, probe_score

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
