import numpy as np
from tqdm import tqdm
from pathlib import Path
import scipy
from metrics import compute_detection_and_identification_rate
import confidence_functions
from multiprocessing import Pool


class TcmNN:
    def __init__(self, number_of_nearest_neighbors, scale, p_value_cache_path) -> None:
        """
        implemets knn based open-set face identification algorithm.
        See
        https://ieeexplore.ieee.org/document/1512050
        and
        https://link.springer.com/chapter/10.1007/3-540-36755-1_32
        """
        self.number_of_nearest_neighbors = number_of_nearest_neighbors
        self.scale = scale  # neaded because we have |D_i^y| = 1 and |D_i^{-y}|!=1
        self.p_value_cache_path = Path(p_value_cache_path)

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

        # 1. compute distances from each gallery class to other gallery classes
        # here each class has exact one feature vector

        gallery_distance_matrix = (
            -np.dot(gallery_feats, gallery_feats.T) + 1
        )  # (3531, 3531)
        D_minus_y = np.sort(gallery_distance_matrix, axis=1)[
            :, 1 : self.number_of_nearest_neighbors + 1
        ]
        D_minus_sum = np.sum(D_minus_y, axis=1)
        # 2. compute distances from each probe feature to all gallery classes
        probe_gallery_distance_matrix = (
            -np.dot(probe_feats, gallery_feats.T) + 1
        )  # (19593, 3531)
        probe_gallery_distance_matrix_sorted = np.argsort(
            probe_gallery_distance_matrix, axis=1
        )[:, : self.number_of_nearest_neighbors + 1]

        # 3. calculate strangeness of gallery samples depending on class of probe sample

        probe_p_values = []
        gallery_strangeness = np.zeros(
            shape=(gallery_feats.shape[0], gallery_feats.shape[0])
        )
        gallery_strangeness += self.scale / D_minus_sum[np.newaxis, :]
        l = gallery_feats.shape[0]
        # here is error as we do not account for just added probe 'probe_index'
        # while computing strangeness for non probe class gallery classes
        cache_path = Path(
            self.p_value_cache_path
            / f"k_{self.number_of_nearest_neighbors}_scale_{self.scale}_gallery_size_{l}.npy"
        )
        if cache_path.is_file():
            probe_p_values = np.load(cache_path)
        else:
            for probe_index in tqdm(range(probe_feats.shape[0])):
                np.fill_diagonal(
                    gallery_strangeness,
                    probe_gallery_distance_matrix[probe_index] / D_minus_sum,
                )
                other_class_distance_sum = []
                default_sum = np.sum(
                    probe_gallery_distance_matrix_sorted[probe_index][:-1]
                )

                for gallery_id in range(gallery_feats.shape[0]):
                    if gallery_id in probe_gallery_distance_matrix_sorted[probe_index]:
                        a = probe_gallery_distance_matrix_sorted[probe_index].copy()
                        a[np.where(a == gallery_id)] = 0
                        other_class_distance_sum.append(np.sum(a))
                    else:
                        other_class_distance_sum.append(default_sum)
                probe_strangeness = probe_gallery_distance_matrix[
                    probe_index
                ] / np.array(other_class_distance_sum)
                # Eq. (8) https://ieeexplore.ieee.org/document/1512050

                p_values = 1 / (l + 1) + (np.sum(gallery_strangeness, axis=1)) / (
                    (l + 1) * probe_strangeness
                )
                probe_p_values.append(p_values)
            probe_p_values = np.array(probe_p_values)  # (19593, 1772)
            np.save(cache_path, probe_p_values)

        similarity = probe_p_values

        p_value_argmax = np.argmax(similarity, axis=1)
        probes_psr = []
        for probe_index in tqdm(range(probe_feats.shape[0])):
            max_idx = p_value_argmax[probe_index]
            a = np.concatenate(
                [
                    similarity[probe_index, :max_idx],
                    similarity[probe_index, max_idx + 1 :],
                ]
            )
            probes_psr.append(
                (similarity[probe_index, max_idx] - np.mean(a)) / np.std(a)
            )

        # BUG: возможно есть ошибка при использовании probes_psr
        top_1_count, top_5_count, top_10_count = 0, 0, 0
        pos_sims, pos_psr, neg_sims, non_gallery_sims, neg_psr = [], [], [], [], []
        for index, query_id in enumerate(probe_ids):
            if query_id in gallery_ids:
                gallery_label = np.argwhere(gallery_ids == query_id)[0, 0]
                index_sorted = np.argsort(similarity[index])[::-1]

                top_1_count += gallery_label in index_sorted[:1]
                top_5_count += gallery_label in index_sorted[:5]
                top_10_count += gallery_label in index_sorted[:10]

                pos_sims.append(similarity[index][gallery_ids == query_id][0])
                pos_psr.append(probes_psr[index])
                neg_sims.append(similarity[index][gallery_ids != query_id])
            else:
                non_gallery_sims.append(similarity[index])
                neg_psr.append(probes_psr[index])
        pos_sims, neg_sims, non_gallery_sims = (
            np.array(pos_sims),
            np.array(neg_sims),
            np.array(non_gallery_sims),
        )
        correct_pos_cond = pos_sims > neg_sims.max(1)
        neg_psr_sorted = np.sort(neg_psr)[::-1]
        threshes, recalls = [], []
        for far in fars:
            thresh = neg_psr_sorted[max(int((neg_psr_sorted.shape[0]) * far) - 1, 0)]
            recall = (
                np.logical_and(correct_pos_cond, pos_psr > thresh).sum()
                / pos_sims.shape[0]
            )
            threshes.append(thresh)
            recalls.append(recall)
        cmc_scores = list(zip(neg_sims, pos_sims.reshape(-1, 1))) + list(
            zip(non_gallery_sims, [None] * non_gallery_sims.shape[0])
        )
        return top_1_count, top_5_count, top_10_count, threshes, recalls, cmc_scores


class EVM:
    def __init__(self, confidence_function_name: str) -> None:
        """
        Implemetns Extreme Value Machine (EVM) and uses it for open set recognition
        in case of one sample of each known class. In particular, we do not perform
        Model Reduction, decried in section IV. A

        https://arxiv.org/abs/1506.06112
        """
        self.confidence_function_name = confidence_function_name

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
        pass


class CosineSim:
    def __init__(self, confidence_function: dict) -> None:
        self.confidence_function = confidence_function

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

        # compute confidences
        confidence_function = getattr(
            confidence_functions, self.confidence_function.class_name
        )(**self.confidence_function.init_args)
        probe_score = confidence_function(similarity)

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


class SCF:
    def __init__(self, confidence_function: dict, k_shift: float) -> None:
        """
        Implements SCF mutual “likelihood” of distributions belonging to the same person (sharing the same latent code)

        https://ieeexplore.ieee.org/document/9577756
        Eq. (13)
        """
        self.confidence_function = confidence_function
        self.k_shift = k_shift

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
        gallery_unc = gallery_unc[np.newaxis, :, 0]

        gallery_unc = gallery_unc + self.k_shift
        probe_unc = probe_unc + self.k_shift

        d = probe_feats.shape[1]
        k_i_times_k_j = probe_unc * gallery_unc
        mu_ij = 2 * np.dot(probe_feats, gallery_feats.T)
        k_ij = np.sqrt(probe_unc**2 + gallery_unc**2 + mu_ij * k_i_times_k_j)

        log_iv_i = (
            np.log(
                1e-6 + scipy.special.ive(d / 2 - 1, probe_unc, dtype=probe_unc.dtype)
            )
            + probe_unc
        )
        log_iv_j = (
            np.log(
                1e-6
                + scipy.special.ive(d / 2 - 1, gallery_unc, dtype=gallery_unc.dtype)
            )
            + gallery_unc
        )
        log_iv_ij = (
            np.log(1e-6 + scipy.special.ive(d / 2 - 1, k_ij, dtype=k_ij.dtype)) + k_ij
        )

        scf_similarity = (
            (d / 2 - 1) * (np.log(probe_unc) + np.log(gallery_unc) - np.log(k_ij))
            - (log_iv_i + log_iv_j - log_iv_ij)
            - d / 2 * np.log(2 * np.pi)
            - d * np.log(64)
        )
        # scf_similarity = -scf_similarity
        similarity = mu_ij / 2
        # similarity = -similarity
        # compute confidences
        confidence_function = getattr(
            confidence_functions, self.confidence_function.class_name
        )(**self.confidence_function.init_args)
        probe_score = confidence_function(scf_similarity)

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


def compute_pfe(
    pfe_similarity, d, probe_feats, probe_sigma_sq, gallery_feats, gallery_sigma_sq
):
    sigma_sq_sum = probe_sigma_sq[:, :, d] + gallery_sigma_sq[:, :, d]
    slice = (
        probe_feats[:, :, d] - gallery_feats[:, :, d]
    ) ** 2 / sigma_sq_sum + np.log(sigma_sq_sum)
    pfe_similarity += slice


class PFE:
    def __init__(self, confidence_function: dict, variance_scale: float) -> None:
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
            # pfe_arguments = [(pfe_similarity, d, probe_feats, probe_unc, gallery_feats, gallery_unc) for d in range(probe_feats.shape[2])]

            for d in tqdm(range(probe_feats.shape[2])):
                compute_pfe(
                    pfe_similarity,
                    d,
                    probe_feats,
                    probe_sigma_sq,
                    gallery_feats,
                    gallery_sigma_sq,
                )
            pfe_similarity = -0.5 * pfe_similarity
            np.save(pfe_cache_path, pfe_similarity)

        # compute confidences
        confidence_function = getattr(
            confidence_functions, self.confidence_function.class_name
        )(**self.confidence_function.init_args)
        probe_score = confidence_function(pfe_similarity)

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
