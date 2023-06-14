import numpy as np

# from evaluation.metrics import compute_detection_and_identification_rate
from evaluation.eval_functions.open_set_identification.abc import Abstract1NEval
from evaluation.confidence_functions import AbstractConfidence
from evaluation.eval_functions.distaince_functions import CosineSimDistance




class CosineSim(Abstract1NEval):
    def __init__(self, confidence_function: AbstractConfidence) -> None:
        self.confidence_function = confidence_function
        self.compute_cosine_sim = CosineSimDistance()
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
        similarity = self.compute_cosine_sim(
            probe_feats, gallery_feats
        )  # np.dot(probe_feats, gallery_feats.T)  # (19593, 1772)
        probe_score = self.confidence_function(similarity)

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
