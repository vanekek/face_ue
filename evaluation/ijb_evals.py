#!/usr/bin/env python3
import os
import numpy as np
from pathlib import Path
import hydra
from hydra.utils import instantiate
import sys

from evaluation.face_recognition_test import Face_Fecognition_test
from evaluation.visualize import (
    plot_dir_far_scores,
    plot_tar_far_scores,
    plot_cmc_scores,
    plot_rejection_scores,
)

path = str(Path(__file__).parent.parent.absolute())
sys.path.insert(1, path)


def instantiate_list(query_list):
    return [instantiate(value) for value in query_list]


@hydra.main(
    config_path=str(
        Path(__file__).resolve().parents[1] / "configs/uncertainty_benchmark"
    ),
    config_name=Path(__file__).stem,
    version_base="1.2",
)
def main(cfg):
    # instantiate
    open_set_identification_metrics = instantiate_list(
        cfg.open_set_identification_metrics
    )
    open_set_uncertainty_metrics = instantiate_list(cfg.open_set_uncertainty_metrics)
    closed_set_identification_metrics = instantiate_list(
        cfg.closed_set_identification_metrics
    )
    verification_metrics = instantiate_list(cfg.verification_metrics)


    test_dataset = instantiate(cfg.test_dataset)

    verif_scores, verif_names = [], []
    open_set_ident_scores, open_set_ident_names = [], []
    closed_set_ident_scores, closed_set_ident_names = [], []

    open_set_ident_rejection_scores, open_set_ident_rejection_names = [], []

    # methods = cfg.open_set_identification_methods + cfg.verification_methods
    # method_types = ["open_set_identification"] * len(
    #     cfg.open_set_identification_methods
    # ) + ["verification"] * len(cfg.verification_methods)
    methods = (
        cfg.closed_set_identification_methods
        + cfg.verification_methods
        + cfg.open_set_identification_methods
    )
    method_types = (
        ["closed_set_identification"] * len(cfg.closed_set_identification_methods)
        + ["verification"] * len(cfg.verification_methods)
        + ["open_set_identification"] * len(cfg.open_set_identification_methods)
    )
    for method, method_type in zip(methods, method_types):
        evaluation_function = instantiate(method.evaluation_function)

        if hasattr(evaluation_function, "__name__"):
            save_name = evaluation_function.__name__
        else:
            save_name = os.path.splitext(os.path.basename(method.save_result))[0]

        template_pooling = instantiate(method.template_pooling_strategy)
        tt = Face_Fecognition_test(
            evaluation_function=evaluation_function,
            test_dataset=test_dataset,
            embeddings_path=method.embeddings_path,
            template_pooling_strategy=template_pooling,
            use_detector_score=method.use_detector_score,
            use_two_galleries=cfg.use_two_galleries,
            recompute_template_pooling=cfg.recompute_template_pooling,
            open_set_identification_metrics=open_set_identification_metrics,
            closed_set_identification_metrics=closed_set_identification_metrics,
            verification_metrics=verification_metrics,
            open_set_uncertainty_metrics=open_set_uncertainty_metrics,
        )

        save_path = os.path.dirname(method.save_result)
        save_items = {}
        if len(save_path) != 0 and not os.path.exists(save_path):
            os.makedirs(save_path)
        if method_type == "open_set_identification":  # 1:N test
            open_set_identification_metric_values = (
                tt.run_model_test_openset_identification()
            )
            fars = open_set_identification_metric_values["fars"]
            open_set_ident_scores.append(
                (fars, open_set_identification_metric_values["recalls"])
            )

            open_set_ident_rejection_scores.append(
                (open_set_identification_metric_values['fractions'],
                 open_set_identification_metric_values['auc_mean_dist_unc'])
            )
            open_set_ident_rejection_names.append(save_name)

            open_set_ident_names.append(save_name)
            print(f"{save_name}:")
            for key in open_set_identification_metric_values.keys():
                if "top" in key:
                    print(
                        f"{key}: {round(open_set_identification_metric_values[key],4)}"
                    )
        elif method_type == "verification":  # Basic 1:1 N0D1F1 test
            continue
            verification_metric_values = tt.run_model_test_verification()
            verif_far = verification_metric_values["fars"]
            verif_scores.append([verif_far, verification_metric_values["recalls"]])
            verif_names.append(save_name)
        elif method_type == "closed_set_identification":
            continue
            closed_set_identification_metric_values = (
                tt.run_model_test_closedset_identification()
            )
            closed_set_ident_scores.append(
                [
                    closed_set_identification_metric_values["ranks"],
                    closed_set_identification_metric_values["cmc"],
                ]
            )
            closed_set_ident_names.append(save_name)
        else:
            raise ValueError
        np.savez(os.path.join(save_path, save_name + ".npz"), **save_items)
    # identif plot
    fig = plot_dir_far_scores(scores=open_set_ident_scores, names=open_set_ident_names)
    fig.savefig(Path(cfg.exp_dir) / "di_far_plot.png", dpi=300)
    print("Plot open ident path:")
    print(str(Path(cfg.exp_dir) / "di_far_plot.png"))

    # verif plot

    fig_verif = plot_tar_far_scores(scores=verif_scores, names=verif_names)
    fig_verif.savefig(Path(cfg.exp_dir) / "tar_far_plot.png", dpi=300)
    print("Plot verif path:")
    print(str(Path(cfg.exp_dir) / "tar_far_plot.png"))

    # cmc plot

    fig_verif = plot_cmc_scores(
        scores=closed_set_ident_scores, names=closed_set_ident_names
    )
    fig_verif.savefig(Path(cfg.exp_dir) / "cmc_plot.png", dpi=300)
    print("Plot closed ident path:")
    print(str(Path(cfg.exp_dir) / "cmc_plot.png"))

    # rejection plot

    fig_rejection = plot_rejection_scores(
        scores=open_set_ident_rejection_scores, names=open_set_ident_rejection_names
    )
    fig_rejection.savefig(Path(cfg.exp_dir) / "rejection_plot.png", dpi=300)
    print("Plot open ident rejection path:")
    print(str(Path(cfg.exp_dir) / "rejection_plot.png"))

if __name__ == "__main__":
    main()
