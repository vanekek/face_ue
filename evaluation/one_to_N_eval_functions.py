import numpy as np
from tqdm import tqdm
from pathlib import Path

class TcmNN:
    def __init__(self, number_of_nearest_neighbors) -> None:
        """
        implemets knn based open-set face identification algorithm.
        See
        https://ieeexplore.ieee.org/document/1512050
        and
        https://link.springer.com/chapter/10.1007/3-540-36755-1_32
        """
        self.number_of_nearest_neighbors = number_of_nearest_neighbors
        self.scale = 0.1  # neaded because we have |D_i^y| = 1 and |D_i^{-y}|!=1

    def __call__(self, query_feats, gallery_feats, query_ids, gallery_ids, fars):
        print(
            "query_feats: %s, gallery_feats: %s"
            % (query_feats.shape, gallery_feats.shape)
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
            -np.dot(query_feats, gallery_feats.T) + 1
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
        cache_path = Path(f'/app/cache/knn_gallery_size_{l}.npy')
        if cache_path.is_file():
            probe_p_values = np.load(cache_path)
        else:
            for probe_index in tqdm(range(query_feats.shape[0])):
                np.fill_diagonal(
                    gallery_strangeness,
                    probe_gallery_distance_matrix[probe_index] / D_minus_sum,
                )
                other_class_distance_sum = []
                default_sum = np.sum(probe_gallery_distance_matrix_sorted[probe_index][:-1])

                for gallery_id in range(gallery_feats.shape[0]):
                    if gallery_id in probe_gallery_distance_matrix_sorted[probe_index]:
                        a = probe_gallery_distance_matrix_sorted[probe_index].copy()
                        a[np.where(a == gallery_id)] = 0
                        other_class_distance_sum.append(np.sum(a))
                    else:
                        other_class_distance_sum.append(default_sum)
                probe_strangeness = probe_gallery_distance_matrix[probe_index] / np.array(
                    other_class_distance_sum
                )
                # Eq. (8) https://ieeexplore.ieee.org/document/1512050
                

                p_values = 1 / (l + 1) + (np.sum(gallery_strangeness, axis=1)) / (
                    (l + 1) * probe_strangeness
                )
                probe_p_values.append(p_values)
            probe_p_values = np.array(probe_p_values)# (19593, 1772)
            np.save(cache_path, probe_p_values)
        
        p_value_argmax = np.argmax(probe_p_values, axis=1)
        probes_psr = []
        for probe_index in tqdm(range(query_feats.shape[0])):
            max_idx = p_value_argmax[probe_index]
            a = np.concatenate([probe_p_values[probe_index,:max_idx], probe_p_values[probe_index, max_idx+1:]])
            probes_psr.append((probe_p_values[probe_index, max_idx] - np.mean(a)) / np.std(a))
        similarity = probe_p_values

        top_1_count, top_5_count, top_10_count = 0, 0, 0
        pos_sims, pos_psr, neg_sims, non_gallery_sims, neg_psr = [], [], [], [], []
        for index, query_id in enumerate(query_ids):
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
            thresh = neg_psr_sorted[
                max(int((neg_psr_sorted.shape[0]) * far) - 1, 0)
            ]
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
    


class PairwiseSims:
    def __init__(self, foo) -> None:
        self.foo = foo

    def __call__(self, query_feats, gallery_feats, query_ids, gallery_ids, fars):
        print(
            "query_feats: %s, gallery_feats: %s"
            % (query_feats.shape, gallery_feats.shape)
        )
        similarity = np.dot(query_feats, gallery_feats.T)  # (19593, 1772)

        top_1_count, top_5_count, top_10_count = 0, 0, 0
        pos_sims, neg_sims, non_gallery_sims = [], [], []
        for index, query_id in enumerate(query_ids):
            if query_id in gallery_ids:
                gallery_label = np.argwhere(gallery_ids == query_id)[0, 0]
                index_sorted = np.argsort(similarity[index])[::-1]

                top_1_count += gallery_label in index_sorted[:1]
                top_5_count += gallery_label in index_sorted[:5]
                top_10_count += gallery_label in index_sorted[:10]

                pos_sims.append(similarity[index][gallery_ids == query_id][0])
                neg_sims.append(similarity[index][gallery_ids != query_id])
            else:
                non_gallery_sims.append(similarity[index])
        total_pos = len(pos_sims)
        pos_sims, neg_sims, non_gallery_sims = (
            np.array(pos_sims),
            np.array(neg_sims),
            np.array(non_gallery_sims),
        )
        print(
            "pos_sims: %s, neg_sims: %s, non_gallery_sims: %s"
            % (pos_sims.shape, neg_sims.shape, non_gallery_sims.shape)
        )
        print(
            "top1: %f, top5: %f, top10: %f"
            % (
                top_1_count / total_pos,
                top_5_count / total_pos,
                top_10_count / total_pos,
            )
        )

        correct_pos_cond = pos_sims > neg_sims.max(1)
        non_gallery_sims_sorted = np.sort(non_gallery_sims.max(1))[::-1]
        threshes, recalls = [], []
        for far in fars:
            thresh = non_gallery_sims_sorted[
                max(int((non_gallery_sims_sorted.shape[0]) * far) - 1, 0)
            ]
            recall = (
                np.logical_and(correct_pos_cond, pos_sims > thresh).sum()
                / pos_sims.shape[0]
            )
            threshes.append(thresh)
            recalls.append(recall)
        cmc_scores = list(zip(neg_sims, pos_sims.reshape(-1, 1))) + list(
            zip(non_gallery_sims, [None] * non_gallery_sims.shape[0])
        )
        return top_1_count, top_5_count, top_10_count, threshes, recalls, cmc_scores
