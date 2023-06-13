import numpy as np

# from evaluation.metrics import compute_detection_and_identification_rate
from evaluation.eval_functions.open_set_identification.abc import Abstract1NEval
from evaluation.confidence_functions import AbstractConfidence


from evaluation.eval_functions.distaince_functions import ScfSim


class SCF(Abstract1NEval):
    def __init__(
        self,
        confidence_function: AbstractConfidence,
        k_shift: float,
        use_cosine_sim_match: bool,
    ) -> None:
        """
        Implements SCF mutual “likelihood” of distributions belonging to the same person (sharing the same latent code)

        https://ieeexplore.ieee.org/document/9577756
        Eq. (13)
        """
        self.confidence_function = confidence_function
        self.k_shift = k_shift
        self.use_cosine_sim_match = use_cosine_sim_match
        self.compute_scf_sim = ScfSim()

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

        scf_similarity = self.compute_scf_sim(
            probe_feats, gallery_feats, gallery_unc, probe_unc
        )

        if self.use_cosine_sim_match:
            similarity = np.dot(probe_feats, gallery_feats.T)
        else:
            similarity = scf_similarity

        # compute confidences
        probe_score = self.confidence_function(scf_similarity)

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
