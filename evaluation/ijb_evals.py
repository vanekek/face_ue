#!/usr/bin/env python3

from pathlib import Path
import hydra
from hydra.utils import instantiate
import sys
import pandas as pd
import numpy as np

from evaluation.face_recognition_test import Face_Fecognition_test
from evaluation.visualize import (
    plot_dir_far_scores,
    plot_cmc_scores,
    plot_rejection_scores,
)

def instantiate_list(query_list):
    return [instantiate(value) for value in query_list]

def get_args_string(d):
    args = []
    for key, value in d.items():
        args.append(f"{key}:{value}")
    return "-".join(args)

def create_method_name(method, sampler, template_pooling, evaluation_function):
    method_name_parts = []
    method_name_parts.append(
        f"sampler-{sampler.__class__.__name__}-num-samples-{sampler.num_samples}"
    )
    method_name_parts.append(f"pooling-with-{template_pooling.__class__.__name__}")
    method_name_parts.append(f"use-det-score-{method.use_detector_score}")
    method_name_parts.append(f"eval-with-{evaluation_function.__class__.__name__}")
    confidence_function = evaluation_function.__dict__["confidence_function"]
    evaluation_function_args = dict(evaluation_function.__dict__)
    evaluation_function_args.pop("confidence_function")
    eval_args = get_args_string(evaluation_function_args)
    if len(eval_args) != 0:
        method_name_parts.append(f"eval-args-{eval_args}")
    method_name_parts.append(f"conf-func-{confidence_function.__class__.__name__}")
    conf_args = get_args_string(confidence_function.__dict__)
    if len(conf_args) != 0:
        method_name_parts.append(f"conf-args-{conf_args}")

    method_name = "_".join(method_name_parts)
    return method_name


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
    if "open_set_uncertainty_metrics" in cfg:
        open_set_uncertainty_metrics = instantiate_list(
            cfg.open_set_uncertainty_metrics
        )
    else:
        open_set_uncertainty_metrics = []
    closed_set_identification_metrics = instantiate_list(
        cfg.closed_set_identification_metrics
    )
    verification_metrics = instantiate_list(cfg.verification_metrics)

    test_dataset = instantiate(cfg.test_dataset)

    verif_scores, verif_names = [], []
    open_set_ident_scores, open_set_ident_names = [], []
    closed_set_ident_scores, closed_set_ident_names = [], []

    open_set_ident_rejection_scores, open_set_ident_rejection_names = [], []

    # create result dirs:
    dataset_name = cfg.test_dataset.dataset_name
    open_set_identification_result_dir = (
        Path(cfg.exp_dir) / dataset_name / "open_set_identification"
    )
    open_set_identification_result_dir.mkdir(exist_ok=True, parents=True)

    closed_set_identification_result_dir = (
        Path(cfg.exp_dir) / dataset_name / "closed_set_identification"
    )
    closed_set_identification_result_dir.mkdir(exist_ok=True, parents=True)

    verification_result_dir = Path(cfg.exp_dir) / dataset_name / "verification"
    verification_result_dir.mkdir(exist_ok=True, parents=True)

    # create result tables place holders
    open_set_recognition_result_metrics = {}
    open_set_uncertainty_result_metrics = {}

    closed_set_recognition_result_metrics = {}
    closed_set_uncertainty_result_metrics = {}

    verification_recognition_result_metrics = {}
    verification_uncertainty_result_metrics = {}

    open_set_ident_pretty_names = {}
    closed_set_ident_pretty_names = {}
    verication_pretty_names = {}
    # define methods

    methods = []
    method_types = []
    if "open_set_identification_methods" in cfg:
        methods += cfg.open_set_identification_methods
        method_types += ["open_set_identification"] * len(
            cfg.open_set_identification_methods
        )
    if "closed_set_identification_methods" in cfg:
        methods += cfg.closed_set_identification_methods
        method_types += ["closed_set_identification"] * len(
            cfg.closed_set_identification_methods
        )
    if "verification_methods" in cfg:
        methods += cfg.verification_methods
        method_types += ["verification"] * len(cfg.verification_methods)

    for method, method_type in zip(methods, method_types):
        sampler = instantiate(method.sampler)
        evaluation_function = instantiate(method.evaluation_function)
        assert evaluation_function is not None
        # if cfg.test_dataset.dataset_name == "survFace" and method.use_detector_score:
        #     continue
        template_pooling = instantiate(method.template_pooling_strategy)
        tt = Face_Fecognition_test(
            sampler=sampler,
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

        if method_type == "open_set_identification":  # 1:N test
            # introduce method name that fully defines method features

            method_name = create_method_name(
                method, sampler, template_pooling, evaluation_function
            )
            print(method_name)
            # run recognition and uncertainty metric computation
            (
                open_set_identification_metric_values,
                open_set_uncertainty_metric_values,
            ) = tt.run_model_test_openset_identification()
            open_set_recognition_result_metrics[
                method_name
            ] = open_set_identification_metric_values

            open_set_uncertainty_result_metrics[
                method_name
            ] = open_set_uncertainty_metric_values

            open_set_ident_pretty_names.update({method_name: method.pretty_name})

        elif method_type == "verification":  # Basic 1:1 N0D1F1 test
            # introduce method name that fully defines method features

            method_name = create_method_name(
                method, sampler, template_pooling, evaluation_function
            )
            print(method_name)
            # run recognition and uncertainty metric computation
            # verification_metric_values, verification_uncertainty_metric_values = tt.run_model_test_verification()
            verification_metric_values = tt.run_model_test_verification()
            verification_recognition_result_metrics[
                method_name
            ] = verification_metric_values

            # verification_uncertainty_result_metrics[
            #     method_name
            # ] = verification_uncertainty_metric_values
            verication_pretty_names.update({method_name: method.pretty_name})
            open_set_ident_pretty_names.update({method_name: method.pretty_name})

        elif method_type == "closed_set_identification":
            method_name = create_method_name(method, template_pooling, evaluation_function)
            print(method_name)

            closed_set_identification_metric_values = (
                tt.run_model_test_closedset_identification()
            )

            closed_set_recognition_result_metrics[
                method_name
            ] = closed_set_identification_metric_values
            closed_set_ident_pretty_names.update({method_name: method.pretty_name})
        else:
            raise ValueError
    # open set identif metric table
    if "open_set_identification_methods" in cfg:
        df = create_open_set_ident_recognition_metric_table(
            open_set_recognition_result_metrics, open_set_ident_pretty_names
        )
        df.to_csv(
            open_set_identification_result_dir / "open_set_identification.csv",
            index=False,
        )
        # save identif plot values
        for method_name in open_set_recognition_result_metrics:
            metrics = open_set_recognition_result_metrics[method_name]
            fars = np.array(metrics["fars"])
            recalls_1_rank = metrics["metric:recalls_1_rank"]
            np.savez(
                open_set_identification_result_dir
                / f"{open_set_ident_pretty_names[method_name]}_recalls_1_rank.npz",
                fars=fars,
                recalls=recalls_1_rank,
            )
        # identif plot
        create_open_set_ident_plots(
            open_set_recognition_result_metrics,
            open_set_identification_result_dir,
            open_set_ident_pretty_names,
        )

        # unc metric table
        df_unc = create_open_set_ident_recognition_metric_table(
            open_set_uncertainty_result_metrics, open_set_ident_pretty_names
        )
        df_unc.to_csv(
            open_set_identification_result_dir / "open_set_unc.csv",
            index=False,
        )
        # unc plot
        create_rejection_plots(
            open_set_uncertainty_result_metrics,
            open_set_identification_result_dir,
            open_set_ident_pretty_names,
        )

    if "verification_methods" in cfg:
        # verification table

        df_verif = create_open_set_ident_recognition_metric_table(
            verification_recognition_result_metrics, verication_pretty_names
        )
        df_verif.to_csv(verification_result_dir / "verification.csv", index=False)
        # # verif plot
    if "closed_set_identification_methods" in cfg:
        # closed set ident table
        df_closed = create_open_set_ident_recognition_metric_table(
            closed_set_recognition_result_metrics, closed_set_ident_pretty_names
        )
        df_closed.to_csv(
            closed_set_identification_result_dir / "closed_set_identification.csv",
            index=False,
        )
        # closed plot
        create_closed_set_ident_plots(
            closed_set_recognition_result_metrics,
            closed_set_identification_result_dir,
            closed_set_ident_pretty_names,
        )


if __name__ == "__main__":
    main()
