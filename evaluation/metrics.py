import numpy as np
from scipy.stats import rankdata
import tqdm


def compute_detection_and_identification_rate(
    fars: np.ndarray,
    probe_ids: np.ndarray,
    gallery_ids: np.ndarray,
    similarity: np.ndarray,
    probe_score: np.ndarray,
):
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
    :return: Detection & identification (DI) rate at each FAR
    """

    top_1_count, top_5_count, top_10_count = 0, 0, 0
    pos_sims, pos_score, neg_sims, non_gallery_sims, neg_score = [], [], [], [], []
    gallery_ranks = rankdata(gallery_ids) - 1
    for index, query_id in enumerate(probe_ids):
        if (gallery_index := np.argwhere(gallery_ids == query_id)):
            # gallery test image

            index_sorted = np.argsort(similarity[index])[::-1]
            rank = gallery_ranks[gallery_index]

            top_1_count += rank in index_sorted[:1]
            top_5_count += rank in index_sorted[:5]
            top_10_count += rank in index_sorted[:10]

            # closeness of test image to true class in gallery
            pos_sims.append(similarity[index][gallery_ids == query_id][0])
            # confidence score of test image classification. If it's low, then
            # this gallery test image gets rejected, lowering the DI rate
            pos_score.append(probe_score[index])
            # closeness of test image to wrong classes in gallery
            neg_sims.append(similarity[index][gallery_ids != query_id])
        else:
            # imposter test image

            # closeness of imposter test image to classes in gallery
            non_gallery_sims.append(similarity[index])
            # confidence score of imposter test image classification. If it's high, then
            # this gallery test image gets accepted, rasing the FAR rate at given operating threshold τ
            neg_score.append(probe_score[index])

    pos_sims, neg_sims, non_gallery_sims = (
        np.array(pos_sims),
        np.array(neg_sims),
        np.array(non_gallery_sims),
    )
    # see which test gallery images have higher closeness to true class in gallery than
    # to the wrong classes
    correct_pos_cond = pos_sims > np.max(neg_sims, axis=1)

    neg_score_sorted = np.sort(neg_score)[::-1]
    threshes, recalls = [], []
    for far in fars:
        # compute operating threshold τ, which gives neaded far
        thresh = neg_score_sorted[max(int((neg_score_sorted.shape[0]) * far) - 1, 0)]

        # compute DI rate at given operating threshold τ
        recall = (
            np.sum(np.logical_and(correct_pos_cond, pos_score > thresh))
            / pos_sims.shape[0]
        )
        threshes.append(thresh)
        recalls.append(recall)

    cmc_scores = list(zip(neg_sims, pos_sims.reshape(-1, 1))) + list(
        zip(non_gallery_sims, [None] * non_gallery_sims.shape[0])
    )

    return top_1_count, top_5_count, top_10_count, threshes, recalls, cmc_scores


def verification_11(
    template_norm_feats=None, unique_templates=None, p1=None, p2=None, batch_size=10000
):
    try:
        print(">>>> Trying cupy.")
        import cupy as cp

        template_norm_feats = cp.array(template_norm_feats)
        score_func = lambda feat1, feat2: cp.sum(feat1 * feat2, axis=-1).get()
        test = score_func(
            template_norm_feats[:batch_size], template_norm_feats[:batch_size]
        )
    except:
        score_func = lambda feat1, feat2: np.sum(feat1 * feat2, -1)

    template2id = np.zeros(max(unique_templates) + 1, dtype=int)
    template2id[unique_templates] = np.arange(len(unique_templates))

    steps = int(np.ceil(len(p1) / batch_size))
    score = []
    for id in tqdm(range(steps), "Verification"):
        feat1 = template_norm_feats[
            template2id[p1[id * batch_size : (id + 1) * batch_size]]
        ]
        feat2 = template_norm_feats[
            template2id[p2[id * batch_size : (id + 1) * batch_size]]
        ]
        score.extend(score_func(feat1, feat2))
    return np.array(score)
