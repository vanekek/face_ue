import numpy as np

from ..metrics import EvalMetricsT, compute_detection_and_identification_rate
from .abc import Abstract1NEval
from .scf import compute_scf_sim


class SVM(Abstract1NEval):
    def __init__(self, use_unc: bool) -> None:
        self.use_unc = use_unc
    def __call__(
        self,
        probe_feats: np.ndarray,
        probe_unc: np.ndarray,
        gallery_feats: np.ndarray,
        gallery_unc: np.ndarray,
        probe_ids: np.ndarray,
        gallery_ids: np.ndarray,
        fars: np.ndarray,
    ) -> EvalMetricsT:
        from sklearn.svm import SVC, LinearSVC
        if self.use_unc:
            XX = compute_scf_sim(
                mu_ij=2 * np.dot(gallery_feats, gallery_feats.T),
                Y_unc=probe_unc,
                X_unc=gallery_unc,
                d=probe_feats.shape[1]
                )
            
            XY = compute_scf_sim(
                mu_ij=2 * np.dot(probe_feats, gallery_feats.T),
                X_unc=gallery_unc,
                Y_unc=probe_unc,
                d=probe_feats.shape[1]
            )
            
            decision_scores = SVC(kernel='precomputed').fit(XX, gallery_ids).decision_function(XY)
        else:
            decision_scores = LinearSVC().fit(gallery_feats, gallery_ids).decision_function(probe_feats)
            
        return compute_detection_and_identification_rate(
            fars,
            probe_ids,
            gallery_ids,
            decision_scores,
            decision_scores.max(1),
            labels_sorted=True
        )
