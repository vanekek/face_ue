import numpy as np
from tqdm import tqdm


def cosine_distance_likelihood(mu, sigma, z):
    z = np.moveaxis(z, 2, 1)  # n x 512 x num_z_samples
    a_ilj_final = mu @ z
    assert a_ilj_final.shape[0] == z.shape[0]
    a_ilj_final = np.moveaxis(a_ilj_final, 2, 1)  # n x num_z_samples x K
    return a_ilj_final

# work's, but has error issues, so predictions are a bit different from default_pfe_likelihood_slow()
def default_pfe_likelihood(mu, sigma, z):
    """
    z: n x num_z_samples x 512
    sigma: K x 512
    mu: K x 512
    """
    b = np.sum(np.log(sigma), axis=1)  # K
    z = np.moveaxis(z, 2, 1)  # n x 512 x num_z_samples

    m = -2 * ((mu / sigma) @ z)  # n x K x num_z_samples
    z_sq = z**2
    t = (1 / sigma) @ z_sq  # n x K x num_z_samples
    n = np.sum(mu**2 / sigma, axis=1)  # K
    a_ilj_final = b[np.newaxis, :, np.newaxis] + t + m + n[np.newaxis, :, np.newaxis]
    a_ilj_final = -0.5 * np.moveaxis(a_ilj_final, 2, 1)
    return a_ilj_final

def default_pfe_likelihood_slow(mu, sigma, z):
    a_ilj_final = np.zeros(shape=z.shape[:-1] + mu.shape[:-1])  # placeholder
    num_dims = mu.shape[1]
    for k in tqdm(range(num_dims)):
        a_ilj = (
            z[:, :, np.newaxis, k] - mu[np.newaxis, np.newaxis, :, k]
        ) ** 2 / sigma[np.newaxis, np.newaxis, :, k] + np.log(sigma)[
            np.newaxis, np.newaxis, :, k
        ]
        np.add(a_ilj_final, a_ilj, out=a_ilj_final)
    return -0.5 * a_ilj_final


def cos_pfe_likelihood(mu, sigma, z):
    """
    z: n x num_z_samples x 512
    sigma: K x 512
    mu: K x 512
    """
    a_ilj_final = np.zeros(shape=z.shape[:-1] + mu.shape[:-1])  # placeholder
    num_dims = mu.shape[1]
    for k in tqdm(range(num_dims)):
        a_ilj = (z[:, :, np.newaxis, k] * mu[np.newaxis, np.newaxis, :, k]) / sigma[
            np.newaxis, np.newaxis, :, k
        ]  # - np.log(sigma)[np.newaxis, np.newaxis, :, k]
        np.add(a_ilj_final, a_ilj, out=a_ilj_final)

    a_ilj_final = 0.5 * a_ilj_final
    return a_ilj_final
