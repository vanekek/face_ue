"""
The script to compare template-image verification protocol, based on IJB-C
It could calculate the PFE, magface, Scale results
Though it requires preprocessed with face detector images
  and pretrained with scripts/train checkpoints
For the options see scripts/eval/template_reject_verification.sh

optimal setups for methods:
magface - mean aggregation, cosine distance, image ue value
PFE - pfe aggregation, mls distance, image ue value
scale - mean aggregation, cosine distance, image ue value
"""


# rewrite the list
# check that pfe ue bigger for verify single images
# correct mls sigma aggregation?


import os
import sys
import numpy as np
from datetime import datetime
import torch
import matplotlib.pyplot as plt
from collections import defaultdict, OrderedDict
from sklearn.metrics import auc
from pathlib import Path
import pickle
from scipy.special import softmax
from scipy.stats import multivariate_normal
from tqdm import tqdm

import hydra

path = str(Path(__file__).parent.parent.parent.absolute())
sys.path.insert(0, path)
from face_lib.datasets import IJBDataset, IJBATest, IJBCTemplates
from face_lib.utils import cfg as cfg_utils
import face_lib.evaluation.plots as plots
from face_lib.evaluation.utils import (
    get_required_models,
    get_distance_uncertainty_funcs,
)
from face_lib.evaluation.feature_extractors import get_features_uncertainties_labels
from face_lib.evaluation.feature_extractors import (
    extract_features_uncertainties_from_list,
)
from face_lib.evaluation.reject_verification import get_rejected_tar_far


from face_lib.evaluation.argument_parser import (
    verify_arguments_template_reject_verification,
    parse_cli_arguments,
)
from face_lib.evaluation.aggregation import aggregate_templates


def compute_probalities(tester, use_mean_z_estimate=True, num_samples=5):
    """
    Computes probability for belonging to each class for each query image

    result of procedure: matrix $P \in \mathbb{R}^{n\times K}$, where n - number of query templates
    K - number of classes.
    Elements are probabilities $p_{ij} = \mathbb{P}(class_{i} = j)$ that i-th query template belongs to j-th class
    """

    z = []  # n x num_samples x 512
    sigma = []  # K x 512
    mu = []  # K x 512
    for query_template in tester.verification_templates():
        z_samples = []
        if use_mean_z_estimate:
            z_samples.append(query_template.mu)
        else:
            pz = multivariate_normal(
                query_template.mu, np.diag(query_template.sigma_sq)
            )
            z_samples.extend(pz.rvs(size=num_samples))
        z.append(z_samples)
    z = np.array(z)
    for enroll_template in tester.enroll_templates():
        sigma.append(enroll_template.sigma_sq)
        mu.append(enroll_template.mu)
    sigma = np.array(sigma)
    mu = np.array(mu)

    # sigma = np.tile(sigma, (z.shape[0], z.shape[1], 1, 1))  # n x num_samples x K x 512
    # mu = np.tile(mu, (z.shape[0], z.shape[1], 1, 1))  # n x num_samples x K x 512
    # z = np.tile(z, (1, 1, mu.shape[2], 1))  # n x num_samples x K x 512
    a_ilj_final = np.zeros(shape=z.shape[:-1] + mu.shape[:-1])  # placeholder
    num_dims = mu.shape[1]
    for k in tqdm(range(num_dims)):
        a_ilj = (
            z[:, :, np.newaxis, k] - mu[np.newaxis, np.newaxis, :, k]
        ) ** 2 / sigma[np.newaxis, np.newaxis, :, k] + np.log(sigma)[
            np.newaxis, np.newaxis, :, k
        ]
        # a_ilj = (
        #     z[:, :, np.newaxis, k] - mu[np.newaxis, np.newaxis, :, k]
        # ) ** 2 / sigma[np.newaxis, np.newaxis, :, k] + np.log(sigma)[
        #     np.newaxis, np.newaxis, :, k
        # ]
        np.add(a_ilj_final, a_ilj, out=a_ilj_final)

    a_ilj_final = -0.5 * a_ilj_final
    p_ij = np.mean(
        softmax(
            a_ilj_final,
            axis=2,
        ),
        axis=1,
    )
    return p_ij


def set_prob_mu(tester, probabilities):
    """
    sets query image mu's to equal to probalitilies to all classes
    """
    for i, query_template in tqdm(enumerate(tester.verification_templates())):
        new_mu = {}
        for j, enroll_template in enumerate(tester.enroll_templates()):
            new_mu[enroll_template.template_id] = probabilities[i, j]
        query_template.mu = new_mu
    for t in tester.enroll_templates():
        t.mu = t.template_id


def set_prob_sigma_sq(tester, probabilities):
    """
    sets query image uncertanties to equal to probalitilies to most likely class
    """

    for i, query_template in enumerate(tester.verification_templates()):
        query_template.sigma_sq = np.max(probabilities[i, :])


def compute_softmax_scores(tester):
    """
    computes softmax scores for all verification_templates

    take first emb from each verification template and computes distances to all enroll templates means
    uncertanty for enroll templates is set to inf and is not used, as we choose min uncertanty agregation
    """

    ver_mus = []
    enroll_mus = []

    for t in tester.verification_templates():
        ver_mus.append(t.mu)

    for t in tester.enroll_templates():
        t.sigma_sq = np.array([np.inf])
        enroll_mus.append(t.mu)

    # compute cosine similarity matrix
    ver_mus = np.array(ver_mus)  # N_ver X embsize
    enroll_mus = np.array(enroll_mus)  # N_enroll X embsize

    sim = ver_mus @ enroll_mus.T  # N_ver X N_enroll

    ver_uncertanty = -np.max(softmax(sim, axis=1), axis=1)

    for t, unc in zip(tester.verification_templates(), ver_uncertanty):
        t.sigma_sq = np.array([unc])


def get_image_embeddings(cfg):
    device = torch.device("cuda:" + str(cfg.device_id))

    model_args = cfg_utils.load_config(cfg.config_path)
    checkpoint = torch.load(cfg.checkpoint_path, map_location=device)

    (
        backbone,
        head,
        discriminator,
        classifier,
        scale_predictor,
        uncertainty_model,
    ) = get_required_models(
        checkpoint=checkpoint, args=cfg, model_args=model_args, device=device
    )
    features_path = Path(cfg.cache_dir) / f"{cfg.uncertainty_strategy}_features.pickle"
    uncertainty_path = (
        Path(cfg.cache_dir) / f"{cfg.uncertainty_strategy}_uncertainty.pickle"
    )

    # Setup the data
    if cfg.protocol != "ijbc":
        raise ValueError('Unkown protocol. Only accept "ijbc" at the moment.')

    testset = IJBDataset(cfg.dataset_path)
    image_paths = testset["abspath"].values
    short_paths = ["/".join(Path(p).parts[-2:]) for p in image_paths]

    if features_path.is_file() and uncertainty_path.is_file():
        with open(features_path, "rb") as f:
            feature_dict = pickle.load(f)
        with open(uncertainty_path, "rb") as f:
            uncertainty_dict = pickle.load(f)
    else:
        print("Calculating")
        if cfg.uncertainty_strategy == "magface":
            raise ValueError("Can't compute magface here")

        if cfg.uncertainty_strategy == "scale_finetuned":
            strategy = "scale"
        else:
            strategy = cfg.uncertainty_strategy

        features, uncertainties = extract_features_uncertainties_from_list(
            backbone,
            head,
            image_paths=image_paths,
            uncertainty_strategy=strategy,
            batch_size=cfg.batch_size,
            discriminator=discriminator,
            scale_predictor=scale_predictor,
            uncertainty_model=uncertainty_model,
            device=device,
            verbose=cfg.verbose,
        )
        feature_dict = {p: feature for p, feature in zip(short_paths, features)}
        with open(features_path, "wb") as f:
            pickle.dump(feature_dict, f)
        uncertainty_dict = {p: scale for p, scale in zip(short_paths, uncertainties)}
        with open(uncertainty_path, "wb") as f:
            pickle.dump(uncertainty_dict, f)
    return image_paths, feature_dict, uncertainty_dict, classifier, device


def set_probability_based_uncertainty(
    tester, cache_path, fusion_name, distance_name, uncertainty_name
):
    # cache probability matrix
    prob_cache_path = Path(cache_path) / f"{fusion_name}_probabilities.npy"
    if prob_cache_path.is_file():
        print("Using cached probabily matrix")
        probabilities = np.load(prob_cache_path)
    else:
        print("Computing probabilities")
        probabilities = compute_probalities(tester)
        np.save(prob_cache_path, probabilities)

    if distance_name == "prob-distance":
        # set probability distances
        print("Setting prob distance")
        set_prob_mu(tester, probabilities)
    if uncertainty_name == "prob-unc":
        print("Setting prob uncertainty")
        # set sigma_sq to be 1 - prob
        set_prob_sigma_sq(tester, probabilities)


def save_plots(
    cfg, all_results, res_AUCs, rejected_portions, distance_fig, uncertainty_fig
):

    for (fusion_name, distance_name, uncertainty_name), aucs in res_AUCs.items():
        print(fusion_name, distance_name, uncertainty_name)
        for FAR, AUC in aucs.items():
            print(f"\tFAR={round(FAR, 5)} TAR_AUC : {round(AUC, 5)}")

    for (
        fusion_name,
        distance_name,
        uncertainty_name,
    ), result_table in all_results.items():
        title = "Template" + distance_name + " " + uncertainty_name
        save_to_path = os.path.join(
            ".",
            fusion_name + "_" + distance_name + "_" + uncertainty_name + ".jpg",
        )

        plots.plot_rejected_TAR_FAR(
            result_table, rejected_portions, title, save_to_path
        )

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    plots.plot_TAR_FAR_different_methods(
        all_results,
        rejected_portions,
        res_AUCs,
        title="Template reject verification",
        save_figs_path=os.path.join(
            ".", f"all_methods_{cfg.uncertainty_strategy}_{timestamp}.jpg"
        ),
    )

    distance_fig.savefig(os.path.join(".", f"distance_dist_{timestamp}.jpg"), dpi=400)
    uncertainty_fig.savefig(
        os.path.join(".", f"uncertainry_dist_{timestamp}.jpg"), dpi=400
    )

    setup_name = {True: "single", False: "full"}[cfg.equal_uncertainty_enroll]

    torch.save(
        all_results,
        os.path.join(
            ".",
            f"table_{cfg.uncertainty_strategy}_{timestamp}_{setup_name}.pt",
        ),
    )


@hydra.main(
    config_path=str(Path(".").resolve() / "configs/hydra"),
    config_name=Path(__file__).stem + "_pfe",
)
def eval_template_reject_verification(cfg):

    rejected_portions = np.linspace(*cfg.rejected_portions)
    FARs = list(map(float, cfg.FARs))
    fusions_distances_uncertainties = list(
        map(lambda x: x.split("_"), cfg.fusion_distance_uncertainty_metrics)
    )

    # Setup the plots
    if rejected_portions is None:
        rejected_portions = [
            0.0,
        ]
    if FARs is None:
        FARs = [
            0.0,
        ]

    all_results = OrderedDict()
    n_figures = len(fusions_distances_uncertainties)
    distance_fig, distance_axes = None, [None] * n_figures
    uncertainty_fig, uncertainty_axes = None, [None] * n_figures

    distance_fig, distance_axes = plt.subplots(
        nrows=1, ncols=n_figures, figsize=(9 * n_figures, 8)
    )
    uncertainty_fig, uncertainty_axes = plt.subplots(
        nrows=1, ncols=n_figures, figsize=(9 * n_figures, 8)
    )
    if n_figures == 1:
        distance_axes = [distance_axes]
        uncertainty_axes = [uncertainty_axes]

    # returns features and uncertainties for a list of images
    (
        image_paths,
        feature_dict,
        uncertainty_dict,
        classifier,
        device,
    ) = get_image_embeddings(cfg)

    tester = IJBCTemplates(image_paths, feature_dict, uncertainty_dict)
    tester.init_proto(cfg.protocol_path)

    prev_fusion_name = None
    for (
        (fusion_name, distance_name, uncertainty_name),
        distance_ax,
        uncertainty_ax,
    ) in zip(fusions_distances_uncertainties, distance_axes, uncertainty_axes):
        print(f"==={fusion_name} {distance_name} {uncertainty_name} ===")

        distance_func, uncertainty_func = get_distance_uncertainty_funcs(
            distance_name=distance_name,
            uncertainty_name=uncertainty_name,
            classifier=classifier,
            device=device,
            distaces_batch_size=cfg.distaces_batch_size,
        )

        if fusion_name != prev_fusion_name:
            if cfg.equal_uncertainty_enroll:
                aggregate_templates(tester.enroll_templates(), fusion_name)
                aggregate_templates(tester.verification_templates(), "first")
            else:
                aggregate_templates(tester.all_templates(), fusion_name)

        if distance_name == "prob-distance" or uncertainty_name == "prob-unc":
            set_probability_based_uncertainty(
                tester, cfg.cache_path, fusion_name, distance_name, uncertainty_name
            )

        (
            feat_1,
            feat_2,
            unc_1,
            unc_2,
            label_vec,
        ) = tester.get_features_uncertainties_labels()

        result_table = get_rejected_tar_far(
            feat_1,
            feat_2,
            unc_1,
            unc_2,
            label_vec,
            distance_func=distance_func,
            pair_uncertainty_func=uncertainty_func,
            uncertainty_mode=cfg.uncertainty_mode,
            FARs=FARs,
            distance_ax=distance_ax,
            uncertainty_ax=uncertainty_ax,
            rejected_portions=rejected_portions,
            equal_uncertainty_enroll=cfg.equal_uncertainty_enroll,
        )

        # delete arrays to prevent memory leak
        del feat_1
        del feat_2
        del unc_1
        del unc_2
        del label_vec

        distance_ax.set_title(f"{distance_name} {uncertainty_name}")
        uncertainty_ax.set_title(f"{distance_name} {uncertainty_name}")

        all_results[(fusion_name, distance_name, uncertainty_name)] = result_table
        prev_fusion_name = fusion_name

    res_AUCs = OrderedDict()
    for method, table in all_results.items():
        res_AUCs[method] = {
            far: auc(rejected_portions, TARs) for far, TARs in table.items()
        }

    save_plots(
        cfg, all_results, res_AUCs, rejected_portions, distance_fig, uncertainty_fig
    )


if __name__ == "__main__":
    eval_template_reject_verification()
