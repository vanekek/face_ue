from tqdm import tqdm
import numexpr as ne
import numpy as np


def compute_scf_sim(
    X_1: np.ndarray, X_2: np.ndarray, X_unc: np.ndarray, Y_unc: np.ndarray
):
    from scipy.special import ive

    d = X_1.shape[1]
    mu_ij = 2 * np.dot(X_1, X_2.T)
    X_unc = X_unc[None, :, 0]
    k_i_times_k_j = Y_unc * X_unc
    k_ij = np.sqrt(Y_unc**2 + X_unc**2 + mu_ij * k_i_times_k_j)

    log_iv_i = np.log(1e-6 + ive(d / 2 - 1, Y_unc, dtype=Y_unc.dtype)) + Y_unc
    log_iv_j = np.log(1e-6 + ive(d / 2 - 1, X_unc, dtype=X_unc.dtype)) + X_unc
    log_iv_ij = np.log(1e-6 + ive(d / 2 - 1, k_ij, dtype=k_ij.dtype)) + k_ij

    scf_similarity = (
        (d / 2 - 1) * (np.log(Y_unc) + np.log(X_unc) - np.log(k_ij))  # type: ignore
        - (log_iv_i + log_iv_j - log_iv_ij)
        - d / 2 * np.log(2 * np.pi)
        - d * np.log(64)
    )

    return scf_similarity


def compute_pfe(
    pfe_similarity,
    chunck_slice,
    probe_feats,
    probe_sigma_sq,
    gallery_feats,
    gallery_sigma_sq,
):
    probe_sigma_sq_slice = probe_sigma_sq[:, :, chunck_slice]
    gallery_sigma_sq_slice = gallery_sigma_sq[:, :, chunck_slice]
    probe_feats_slice = probe_feats[:, :, chunck_slice]
    gallery_feats_slice = gallery_feats[:, :, chunck_slice]
    sigma_sq_sum = ne.evaluate("probe_sigma_sq_slice + gallery_sigma_sq_slice")
    slice = ne.evaluate(
        "(probe_feats_slice - gallery_feats_slice)**2 / sigma_sq_sum + log(sigma_sq_sum)"
    )
    slice_sum = ne.evaluate("sum(slice, axis=2)")
    ne.evaluate("slice_sum + pfe_similarity", out=pfe_similarity)


def compute_pfe_sim(
    X: np.ndarray,
    Y: np.ndarray,
    X_unc: np.ndarray,
    Y_unc: np.ndarray,
    pfe_cache_path=None,
):
    X = X[:, np.newaxis, :]
    X_unc = X_unc[:, np.newaxis, :]

    Y = Y[np.newaxis, :, :]
    Y_unc = Y_unc[np.newaxis, :, :]
    pfe_similarity = np.zeros((X.shape[0], Y.shape[0]))

    chunck_size = 128
    for d in tqdm(range(X.shape[2] // chunck_size)):
        compute_pfe(
            pfe_similarity,
            slice(d * chunck_size, (d + 1) * chunck_size),
            X,
            X_unc,
            Y,
            Y_unc,
        )
    pfe_similarity = -0.5 * pfe_similarity
    if pfe_cache_path is not None:
        np.save(pfe_cache_path, pfe_similarity)
    return pfe_similarity
