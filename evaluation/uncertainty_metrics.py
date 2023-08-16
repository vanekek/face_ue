from typing import Any, List, Tuple
import numpy as np
from sklearn.metrics import top_k_accuracy_score
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc
from scipy import interpolate
from evaluation.confidence_functions import MisesProb
import scipy

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
    unc_metrics = {"fractions": fractions}
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
            if "metric:AUC" in key:
                rank = key.split("_")[1]
                auc_res = metric[key]
                unc_metrics[
                    f"final_auc_{rank}_unc_{metric_name}_frac_{np.round(fraction, 3)}"
                ] = auc_res
                auc_metric_name = f"plot_reject_AUCS_{rank}_rank_{metric_name}"
                if auc_metric_name in unc_metrics:
                    unc_metrics[auc_metric_name].append(auc_res)
                else:
                    unc_metrics[auc_metric_name] = [auc_res]
            elif "metric:recall-at-far" in key:
                far = key.split("_")[-3]
                rank = key.split("_")[-2]
                recall_at_far_metric_name = (
                    f"plot_reject_DIR-at-FAR-{far}_{rank}_rank_{metric_name}"
                )
                if recall_at_far_metric_name in unc_metrics:
                    unc_metrics[recall_at_far_metric_name].append(metric[key])
                else:
                    unc_metrics[recall_at_far_metric_name] = [metric[key]]
            elif "error_count::false-rejection-count-at-far" in key:
                far = key.split("_")[-3]
                rank = key.split("_")[-2]
                false_rejection_metric_name = (
                    f"plot_reject_false-rejection-{far}_{rank}_rank_{metric_name}"
                )
                if false_rejection_metric_name in unc_metrics:
                    unc_metrics[false_rejection_metric_name].append(metric[key])
                else:
                    unc_metrics[false_rejection_metric_name] = [metric[key]]
            elif "error_count:false-ident-count" in key:
                rank = key.split("_")[-2]
                false_ident_metric_name = (
                    f"plot_reject_false-ident_{rank}_rank_{metric_name}"
                )
                if false_ident_metric_name in unc_metrics:
                    unc_metrics[false_ident_metric_name].append(metric[key])
                else:
                    unc_metrics[false_ident_metric_name] = [metric[key]]
                
    for key in unc_metrics:
        if "plot_reject" in key:
            unc_metrics[key] = np.array(unc_metrics[key])

    return unc_metrics


# class OptimalRecallReject:
#     def __init__(self, fractions: List[int], metric_to_monitor: Any) -> None:
#         self.fractions = np.arange(fractions[0], fractions[1], step=fractions[2])
#         self.metric_to_monitor = metric_to_monitor

#     def __call__(
#         self,
#         probe_ids: np.ndarray,
#         probe_template_unc: np.ndarray,
#         gallery_ids: np.ndarray,
#         similarity: np.ndarray,
#         probe_score: np.ndarray,
#     ) -> Any:
#         unc_metrics ={"fractions": self.fractions}
#         metric = self.metric_to_monitor(
#             probe_ids=probe_ids,
#             gallery_ids=gallery_ids,
#             similarity=np.mean(similarity, axis=1),
#             probe_score=probe_score,
#         )
#         gallery_ids_argsort = np.argsort(gallery_ids)
#         gallery_ids = gallery_ids[gallery_ids_argsort]


#         # sort labels
#         similarity = similarity[:, gallery_ids_argsort]

#         is_seen = np.isin(probe_ids, gallery_ids)

#         seen_sim: np.ndarray = similarity[is_seen]
#         most_similar_classes = np.argsort(seen_sim, axis=1)[:, ::-1]
#         seen_probe_ids = probe_ids[is_seen]

#         for key, value in metric.items():
#             if "misc:error-num-at-far" in key:
#                 rank = key.split("_")[-2]
#                 far = key.split("_")[-3]
#                 optimal_recall_at_far_metric_name = f"plot_reject_DIR-at-FAR-{far}_{rank}_rank_OptimalRecall"

#                 unc_metrics[optimal_recall_at_far_metric_name] = [metric[key]]


#         return unc_metrics


class DataUncertaintyReject:
    def __init__(
        self, metric_to_monitor: any, fractions: List[int], is_confidence: bool
    ) -> None:
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
        unc_metric_name = self.__class__.__name__
        unc_metric = get_reject_metrics(
            unc_metric_name,
            unc_score,
            self.metric_to_monitor,
            probe_ids,
            gallery_ids,
            np.mean(similarity, axis=1),
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
        unc_metric_name = self.__class__.__name__
        # unc_metric_name = r"$m(p) = \frac{s(p)+1}{2}\left(1 - \frac{s(p)+1}{2}\right)$" #
        unc_metric = get_reject_metrics(
            unc_metric_name,
            unc_score,
            self.metric_to_monitor,
            probe_ids,
            gallery_ids,
            np.mean(similarity, axis=1),
            probe_score,
            self.fractions,
        )
        return unc_metric


class MaxProb:
    def __init__(
        self,
        metric_to_monitor: any,
        fractions: List[int],
    ) -> None:
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
        unc_score = -np.max(similarity, axis=1)
        unc_metric_name = self.__class__.__name__
        # unc_metric_name = r"$m(p) = \max_{c\in {1,\dots,K+1}}p(c|z)$"
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
        use_maxprob_variance: bool,
        data_variance_weight: float,
    ) -> None:
        self.fractions = np.arange(fractions[0], fractions[1], step=fractions[2])
        self.metric_to_monitor = metric_to_monitor
        self.kappa = kappa
        self.beta = beta
        self.use_maxprob_variance = use_maxprob_variance
        self.data_variance_weight = data_variance_weight
        assert self.data_variance_weight >= 0 and self.data_variance_weight <= 1
        self.mises_maxprob = MisesProb(kappa=self.kappa, beta=self.beta)

    def __call__(
        self,
        probe_ids: np.ndarray,
        probe_template_unc: np.ndarray,
        gallery_ids: np.ndarray,
        similarity: np.ndarray,
        probe_score: np.ndarray,
    ) -> Any:
        all_classes_log_prob = self.mises_maxprob.compute_all_class_log_probabilities(
            similarity
        )

        unc_metric_name = (
            (self.__class__.__name__)
            + ",beta="
            + str(self.beta)
            + ",alpha="
            + str(self.data_variance_weight)
        )

        unc_score = -np.mean(np.max(all_classes_log_prob, axis=-1), axis=-1)
        data_uncertainty = -probe_template_unc[:, 0]
        data_uncertainty = (data_uncertainty - np.min(data_uncertainty)) / (
            np.max(data_uncertainty) - np.min(data_uncertainty)
        )
        unc_score = (unc_score - np.min(unc_score)) / (
            np.max(unc_score) - np.min(unc_score)
        )

        unc_score = unc_score * (
            1 - self.data_variance_weight
        ) + self.data_variance_weight * (data_uncertainty)
        # unc_score = 1 - (1-data_uncertainty)*(1-unc_score)

        # unc_metric_name = r"$m(p) = \max_{c\in {1,\dots,K+1}}p(c|z)$"
        unc_metric = get_reject_metrics(
            unc_metric_name,
            unc_score,
            self.metric_to_monitor,
            probe_ids,
            gallery_ids,
            np.mean(similarity, axis=1),
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
        unc_metric_name = self.__class__.__name__
        # unc_metric_name = r"$m(p) = \left|s(p)-\frac{1}{N}\sum_{p\in TestSet}s(p)\right|$" #
        unc_metric = get_reject_metrics(
            unc_metric_name,
            unc_score,
            self.metric_to_monitor,
            probe_ids,
            gallery_ids,
            np.mean(similarity, axis=1),
            probe_score,
            self.fractions,
        )
        return unc_metric
