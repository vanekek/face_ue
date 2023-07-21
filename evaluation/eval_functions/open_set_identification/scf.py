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
        cosine_pred: bool,
    ) -> None:
        """
        Implements SCF mutual “likelihood” of distributions belonging to the same person (sharing the same latent code)

        https://ieeexplore.ieee.org/document/9577756
        Eq. (13)
        """
        self.confidence_function = confidence_function
        self.k_shift = k_shift
        self.cosine_pred = cosine_pred

    def __call__(
        self,
        probe_feats,
        probe_unc,
        gallery_feats,
        gallery_unc,
    ):
        print(
            "probe_feats: %s, gallery_feats: %s"
            % (probe_feats.shape, gallery_feats.shape)
        )
        compute_scf_sim = ScfSim()
        gallery_unc = gallery_unc + self.k_shift
        probe_unc = probe_unc + self.k_shift

        scf_similarity = compute_scf_sim(
            probe_feats, gallery_feats, gallery_unc, probe_unc
        )

        if self.cosine_pred:
            similarity = np.dot(probe_feats, gallery_feats.T)
        else:
            similarity = scf_similarity

        # compute confidences
        probe_score = self.confidence_function(similarity)

        return similarity, probe_score
