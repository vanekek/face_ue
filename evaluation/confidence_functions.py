import numpy as np


def max_similarity_value_confidence(similarity_matrix: np.ndarray):
    """
    Returns confidence for each test image of being gallery (known class) sample
    Here we take similarity to most similar class in gallery as confidence measure

    :param similarity: (probe_size, gallery_size) marix, which specifies closeness of all test images to each gallery class
    :return probe_score: specifies confinence that particular test image belongs to predicted class
        image's probe_score is less than operating threshold τ, then this image get rejected as imposter
    """
    return np.max(similarity_matrix, axis=1)
