from typing import List, Tuple
import numpy as np
from sklearn.metrics import top_k_accuracy_score
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc

EvalMetricsT = Tuple[int, int, int, List[float], List[float], List[Tuple[float, float]]]


class CMC:
    def __init__(self, top_n_ranks: List[int]) -> None:
        self.top_n_ranks = top_n_ranks

    def __call__(
        self,
        probe_ids: np.ndarray,
        gallery_ids: np.ndarray,
        similarity: np.ndarray,
        probe_score: np.ndarray,
    ):
        cmc = []
        most_similar_classes = np.argsort(similarity, axis=1)[:, ::-1]
        for n in self.top_n_ranks:
            n_similart_classes = gallery_ids[most_similar_classes[:, :n]]
            is_seen = np.isin(probe_ids, n_similart_classes)
            cmc.append(np.sum(is_seen) / probe_ids.shape[0])

        metrics = {"ranks": self.top_n_ranks, "cmc": np.array(cmc)}
        return metrics


class TarFar:
    def __init__(self, far_range: List[int]) -> None:
        self.fars = [
            10**ii for ii in np.arange(far_range[0], far_range[1], 4.0 / far_range[2])
        ] + [1]

    def __call__(self, scores, labels):
        true_match_scores = scores[labels == 1]
        wrong_match_scores = scores[labels == 0]

        threshes, recalls = [], []
        wrong_match_scores_sorted = np.sort(wrong_match_scores)[::-1]
        for far in self.fars:
            thresh = wrong_match_scores_sorted[
                max(int((wrong_match_scores_sorted.shape[0]) * far) - 1, 0)
            ]
            recall = np.sum(true_match_scores > thresh) / true_match_scores.shape[0]
            threshes.append(thresh)
            recalls.append(recall)
        metrics = {
            "fars": self.fars,
            "recalls": np.array(recalls),
            "auc": auc(self.fars, np.array(recalls)),
        }
        return metrics


class DetectionAndIdentificationRate:
    def __init__(self, top_n_ranks: List[int], far_range: List[int]) -> None:
        self.top_n_ranks = top_n_ranks
        self.fars = [
            10**ii for ii in np.arange(far_range[0], far_range[1], 4.0 / far_range[2])
        ] + [1]

    def __call__(
        self,
        probe_ids: np.ndarray,
        gallery_ids: np.ndarray,
        similarity: np.ndarray,
        probe_score: np.ndarray,
        labels_sorted: bool = False,
    ) -> EvalMetricsT:
        """
        Computes Detection & identification rate for open set recognition
        Operating thresholds τ for rejecting imposter images are computed to match particular far in fars list
        We assume that for each test image, gallery class with highest similarity is selected as predicted class.
        See
        Handbook of Face Recognition
        https://link.springer.com/book/10.1007/978-0-85729-932-1
        page 554

        :param fars: List of false acceptance rates. Defines proportion of imposter test images,
            which gets wrongly classified as gallery image (i.e match score is above an operating threshold τ)
        :param probe_ids: List of true id's (or classes in general case) of test images
        :param gallery_ids: List of true id's (or classes in general case) of gallery images
        :param similarity: (probe_size, gallery_size) marix, which specifies closeness of all test images to each gallery class
        :param probe_score: specifies confinence that particular test image belongs to predicted class
            image's probe_score is less than operating threshold τ, then this image get rejected as imposter
        :param labels_sorted: specifies the order of labels in similarity matrix.
            If True, assumes the order is ascending, else assumes order is the same as in gallery_ids
        :return: Detection & identification (DI) rate at each FAR
        """
        gallery_ids_argsort = np.argsort(gallery_ids)
        if not labels_sorted:
            similarity = similarity[:, gallery_ids_argsort]

        is_seen = np.isin(probe_ids, gallery_ids)

        seen_sim: np.ndarray = similarity[is_seen]
        seen_probe_ids = probe_ids[is_seen]

        # def topk(k) -> int:
        #     raise NotImplemented
        #     return top_k_accuracy_score(seen_probe_ids, seen_sim, k=k, normalize=False)  # type: ignore

        # top_n_count = map(topk, self.top_n_ranks)

        # Boolean mask (seen_probes, gallery_ids), 1 where the probe matches gallery sample
        pos_mask: np.ndarray = (
            probe_ids[is_seen, None] == gallery_ids[None, gallery_ids_argsort]
        )

        pos_sims = seen_sim[pos_mask]
        neg_sims = seen_sim[~pos_mask].reshape(*pos_sims.shape, -1)
        pos_score = probe_score[is_seen]
        neg_score = probe_score[~is_seen]
        non_gallery_sims = similarity[~is_seen]

        # see which test gallery images have higher closeness to true class in gallery than
        # to the wrong classes
        correct_pos_cond = pos_sims > np.max(neg_sims, axis=1)

        neg_score_sorted = np.sort(neg_score)[::-1]
        threshes, recalls = [], []
        for far in self.fars:
            # compute operating threshold τ, which gives neaded far
            thresh = neg_score_sorted[
                max(int((neg_score_sorted.shape[0]) * far) - 1, 0)
            ]

            # compute DI rate at given operating threshold τ
            recall = (
                np.sum(np.logical_and(correct_pos_cond, pos_score > thresh))
                / pos_sims.shape[0]
            )
            threshes.append(thresh)
            recalls.append(recall)
        recalls = np.array(recalls)

        # metrics = dict(zip([f"top_{k}_count" for k in self.top_n_ranks], top_n_count))
        metrics = {}
        metrics.update({"fars": self.fars, "recalls": recalls})
        return metrics
