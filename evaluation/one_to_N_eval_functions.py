import numpy as np

class TcmNN:
    def __init__(self, number_of_nearest_neighbors) -> None:
        self.number_of_nearest_neighbors = number_of_nearest_neighbors
    def __call__(self, query_feats, gallery_feats, query_ids, gallery_ids, fars):
        print(
            "query_feats: %s, gallery_feats: %s" % (query_feats.shape, gallery_feats.shape)
        )
        similarity = np.dot(query_feats, gallery_feats.T)  # (19593, 3531)        
class PairwiseSims:
    def __init__(self, foo) -> None:
        self.foo = foo

    def __call__(self, query_feats, gallery_feats, query_ids, gallery_ids, fars):
        print(
            "query_feats: %s, gallery_feats: %s" % (query_feats.shape, gallery_feats.shape)
        )
        similarity = np.dot(query_feats, gallery_feats.T)  # (19593, 3531)

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
            % (top_1_count / total_pos, top_5_count / total_pos, top_10_count / total_pos)
        )

        correct_pos_cond = pos_sims > neg_sims.max(1)
        non_gallery_sims_sorted = np.sort(non_gallery_sims.max(1))[::-1]
        threshes, recalls = [], []
        for far in fars:
            # thresh = non_gallery_sims_sorted[int(np.ceil(non_gallery_sims_sorted.shape[0] * far)) - 1]
            thresh = non_gallery_sims_sorted[
                max(int((non_gallery_sims_sorted.shape[0]) * far) - 1, 0)
            ]
            recall = (
                np.logical_and(correct_pos_cond, pos_sims > thresh).sum()
                / pos_sims.shape[0]
            )
            threshes.append(thresh)
            recalls.append(recall)
            # print("FAR = {:.10f} TPIR = {:.10f} th = {:.10f}".format(far, recall, thresh))
        cmc_scores = list(zip(neg_sims, pos_sims.reshape(-1, 1))) + list(
            zip(non_gallery_sims, [None] * non_gallery_sims.shape[0])
        )
        return top_1_count, top_5_count, top_10_count, threshes, recalls, cmc_scores