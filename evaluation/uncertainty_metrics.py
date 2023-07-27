from typing import Any, List, Tuple
import numpy as np
from sklearn.metrics import top_k_accuracy_score
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc
from scipy import interpolate
from evaluation.confidence_functions import MisesProb

EvalMetricsT = Tuple[int, int, int, List[float], List[float], List[Tuple[float, float]]]


def get_reject_metrics(
    metric_name,
    unc_score,
    metric_to_monitor,
    probe_ids,
    gallery_ids,
    similarity,
    probe_score,
    fractions,
):
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
                aucs[
                    f"final_auc_{rank}_unc_{metric_name}_frac_{np.round(fraction, 3)}"
                ] = auc_res
                if f"plot_auc_{rank}_rank_{metric_name}" in aucs:
                    aucs[f"plot_auc_{rank}_rank_{metric_name}"].append(auc_res)
                else:
                    aucs[f"plot_auc_{rank}_rank_{metric_name}"] = [auc_res]

    for key in aucs:
        if "plot_auc_" in key:
            aucs[key] = np.array(aucs[key])
    unc_metric = {"fractions": fractions}
    unc_metric.update(aucs)
    return unc_metric

class OptimalReject:
    def __init__(self, metric_to_monitor: any, fractions: List[int]) -> None:
        self.fractions = np.arange(fractions[0], fractions[1], step=fractions[2])
        self.metric_to_monitor = metric_to_monitor
    
    def __call__(
            self,
            probe_ids: np.ndarray,
            probe_template_unc: np.ndarray,
            gallery_ids: np.ndarray,
            similarity: np.ndarray,
            probe_score: np.ndarray,
        ) -> Any:
        pass

class DataUncertaintyReject:
    def __init__(self, metric_to_monitor: any, fractions: List[int], is_confidence: bool) -> None:
        self.fractions = np.arange(fractions[0], fractions[1], step=fractions[2])
        self.metric_to_monitor = metric_to_monitor
        self.is_confidence = is_confidence
    def __call__(
            self,
            probe_ids: np.ndarray,
            probe_template_unc: np.ndarray,
            gallery_ids: np.ndarray,
            similarity: np.ndarray,
            probe_score: np.ndarray,
        ) -> Any:
        if self.is_confidence:
            unc_score = -probe_template_unc[:, 0]
        else:
            unc_score = probe_template_unc
        unc_metric_name = (
            self.__class__.__name__
        )
        unc_metric = get_reject_metrics(
            unc_metric_name,
            unc_score,
            self.metric_to_monitor,
            probe_ids,
            gallery_ids,
            similarity,
            probe_score,
            self.fractions,
        )
        return unc_metric
class BernoulliVarianceReject:
    def __init__(self, metric_to_monitor: any, fractions: List[int]) -> None:
        self.fractions = np.arange(fractions[0], fractions[1], step=fractions[2])
        self.metric_to_monitor = metric_to_monitor

    def __call__(
        self,
        probe_ids: np.ndarray,
        probe_template_unc: np.ndarray,
        gallery_ids: np.ndarray,
        similarity: np.ndarray,
        probe_score: np.ndarray,
    ) -> Any:
        probe_score_norm = (probe_score + 1) / 2
        unc_score = probe_score_norm * (1 - probe_score_norm)
        unc_metric_name = (
            self.__class__.__name__
        ) 
        #unc_metric_name = r"$m(p) = \frac{s(p)+1}{2}\left(1 - \frac{s(p)+1}{2}\right)$" #
        unc_metric = get_reject_metrics(
            unc_metric_name,
            unc_score,
            self.metric_to_monitor,
            probe_ids,
            gallery_ids,
            similarity,
            probe_score,
            self.fractions,
        )
        return unc_metric


class CombinedMaxProb:
    def __init__(
        self,
        metric_to_monitor: any,
        fractions: List[int],
        kappa: float,
        beta: float,
        with_unc: bool,
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
        all_classes_log_prob = mises_maxprob.compute_all_class_log_probabilities(
            similarity
        )
        unc_score = -np.max(all_classes_log_prob, axis=1)
        unc_metric_name = (
            self.__class__.__name__
        )
        #unc_metric_name = r"$m(p) = \max_{c\in {1,\dots,K+1}}p(c|z)$"
        unc_metric = get_reject_metrics(
            unc_metric_name,
            unc_score,
            self.metric_to_monitor,
            probe_ids,
            gallery_ids,
            similarity,
            probe_score,
            self.fractions,
        )
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
        unc_metric_name = (
            self.__class__.__name__
        ) 
        #unc_metric_name = r"$m(p) = \left|s(p)-\frac{1}{N}\sum_{p\in TestSet}s(p)\right|$" #
        unc_metric = get_reject_metrics(
            unc_metric_name,
            unc_score,
            self.metric_to_monitor,
            probe_ids,
            gallery_ids,
            similarity,
            probe_score,
            self.fractions,
        )
        return unc_metric
