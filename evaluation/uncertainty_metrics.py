from typing import Any, List, Tuple
import numpy as np
from sklearn.metrics import top_k_accuracy_score
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc
from scipy import interpolate
from evaluation.confidence_functions import MisesProb

EvalMetricsT = Tuple[int, int, int, List[float], List[float], List[Tuple[float, float]]]

def get_reject_metrics(unc_score, metric_to_monitor, probe_ids, gallery_ids, similarity, probe_score, fractions):
    unc_indexes = np.argsort(unc_score)
    aucs = {}
    for fraction in fractions:
        # drop worst fraction
        good_probes_idx = unc_indexes[: int((1 - fraction) * probe_ids.shape[0])]
        metric = metric_to_monitor(
            probe_ids=probe_ids[good_probes_idx],
            gallery_ids=gallery_ids,
            similarity=similarity[good_probes_idx],
            probe_score=probe_score[good_probes_idx],
        )
        for key, value in metric.items():
            if "recalls" in key:
                rank = key.split("_")[1]
                auc_res = auc(metric["fars"], metric[key])
                aucs[f"final_auc_{rank}_unc_frac_{np.round(fraction, 3)}"] = auc_res
                if f"plot_auc_{rank}_rank_mean_dist_unc" in aucs:
                    aucs[f"plot_auc_{rank}_rank_mean_dist_unc"].append(auc_res)
                else:
                    aucs[f"plot_auc_{rank}_rank_mean_dist_unc"] = [auc_res]

    for key in aucs:
        if "plot_auc_" in key:
            aucs[key] = np.array(aucs[key])
    unc_metric = {"fractions": fractions}
    unc_metric.update(aucs)
    return unc_metric


class CombinedMaxProb:
    def __init__(
        self, metric_to_monitor: any, fractions: List[int], kappa: float, beta:float, with_unc: bool
    ) -> None:
        self.fractions = np.arange(fractions[0], fractions[1], step=fractions[2])
        self.metric_to_monitor = metric_to_monitor
        self.kappa = kappa
        self.beta = beta
        self.with_unc = with_unc

    def __call__(
        self,
        probe_ids: np.ndarray,
        probe_template_unc: np.ndarray,
        gallery_ids: np.ndarray,
        similarity: np.ndarray,
        probe_score: np.ndarray,
    ) -> Any:
        mises_maxprob = MisesProb(kappa=self.kappa, beta=self.beta)
        all_classes_log_prob = mises_maxprob.compute_all_class_log_probabilities(similarity)
        unc_score = np.max(all_classes_log_prob, axis=1)
        unc_metric = get_reject_metrics(unc_score, self.metric_to_monitor, probe_ids, gallery_ids, similarity, probe_score, self.fractions)
        return unc_metric



class MeanDistanceReject:
    def __init__(
        self, metric_to_monitor: any, fractions: List[int], with_unc: bool
    ) -> None:
        self.fractions = np.arange(fractions[0], fractions[1], step=fractions[2])
        self.metric_to_monitor = metric_to_monitor
        self.with_unc = with_unc

    def __call__(
        self,
        probe_ids: np.ndarray,
        probe_template_unc: np.ndarray,
        gallery_ids: np.ndarray,
        similarity: np.ndarray,
        probe_score: np.ndarray,
    ) -> Any:
        mean_probe_score = np.mean(probe_score)
        unc_score = -np.abs(probe_score - mean_probe_score)

        unc_metric = get_reject_metrics(unc_score, self.metric_to_monitor, probe_ids, gallery_ids, similarity, probe_score, self.fractions)
        return unc_metric
