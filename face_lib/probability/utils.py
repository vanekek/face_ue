from scipy.stats import multivariate_normal
import numpy as np
from scipy.special import softmax
import numpy as np
from scipy.special import softmax
from scipy.stats import multivariate_normal


def compute_probalities(
    tester, likelihood_function, use_mean_z_estimate, num_z_samples
):
    """
    Computes probability for belonging to each class for each query image

    result of procedure: matrix $P \in \mathbb{R}^{n\times K}$, where n - number of query templates
    K - number of classes.
    Elements are probabilities $p_{ij} = \mathbb{P}(class_{i} = j)$ that i-th query template belongs to j-th class
    """

    z = []  # n x num_z_samples x 512
    sigma = []  # K x 512
    mu = []  # K x 512

    # sample z's for each query image
    for query_template in tester.verification_templates():
        z_samples = []
        if use_mean_z_estimate:
            z_samples.append(query_template.mu)
        else:
            pz = multivariate_normal(
                query_template.mu, np.diag(query_template.sigma_sq)
            )
            z_samples.extend(pz.rvs(size=num_z_samples))
        z.append(z_samples)
    z = np.array(z)
    for enroll_template in tester.enroll_templates():
        sigma.append(enroll_template.sigma_sq)
        mu.append(enroll_template.mu)
    sigma = np.array(sigma)
    mu = np.array(mu)

    a_ilj_final = likelihood_function(mu, sigma, z)

    p_ij = np.mean(
        softmax(
            a_ilj_final,
            axis=2,
        ),
        axis=1,
    )
    return p_ij
