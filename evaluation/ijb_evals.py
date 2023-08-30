#!/usr/bin/env python3

from pathlib import Path
import hydra
from hydra.utils import instantiate
import numpy as np
from itertools import product
from evaluation.face_recognition_test import Face_Fecognition_test


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


def init_methods(cfg):
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
    return methods, method_types


@hydra.main(
    config_path=str(
        Path(__file__).resolve().parents[1] / "configs/uncertainty_benchmark"
    ),
    config_name=Path(__file__).stem,
    version_base="1.2",
)
def main(cfg):
    # define methods
    methods, method_task_type = init_methods(cfg)
    tasks_names = list(set(method_task_type))

    # instantiate metrics
    recognition_metrics = {
        task: instantiate_list(getattr(cfg, f"{task}_metrics")) for task in tasks_names
    }
    uncertainty_metrics = {
        task: instantiate_list(getattr(cfg, f"{task}_uncertainty_metrics"))
        for task in tasks_names
    }

    # instantiate datasets
    test_datasets = instantiate_list(cfg.test_dataset)
    dataset_names = [test_dataset.dataset_name for test_dataset in test_datasets]
    # create result dictionary
    metric_values = {
        (task, dataset_name): {"recognition": {}, "uncertainty": {}}
        for task, dataset_name in zip(tasks_names, dataset_names)
    }

    # create pretty name map
    pretty_names = {task: {} for task in tasks_names}

    # run face recognition methods
    for (method, task_type), test_dataset in product(
        zip(methods, method_task_type), test_datasets
    ):
        dataset_name = test_dataset.dataset_name

        # instantiate method
        template_pooling = instantiate(method.template_pooling_strategy)
        sampler = instantiate(method.sampler)
        distance_function = instantiate(method.distance_function)
        recognition_method = instantiate(method.recognition_method)

        # create unique method name
        method_name = create_method_name(
            method, sampler, template_pooling, distance_function
        )
        print(method_name)
        pretty_names[task_type][method_name] = method.pretty_name

        # create tester
        tt = Face_Fecognition_test(
            task_type=task_type,
            method_name=method_name,
            recognition_method=recognition_method,
            sampler=sampler,
            distance_function=distance_function,
            test_dataset=test_dataset,
            embeddings_path=method.embeddings_path,
            template_pooling_strategy=template_pooling,
            use_detector_score=method.use_detector_score,
            use_two_galleries=cfg.use_two_galleries,
            recompute_template_pooling=cfg.recompute_template_pooling,
            recognition_metrics=recognition_metrics,
            uncertainty_metrics=uncertainty_metrics,
        )

        (
            recognition_metric_values,
            uncertainty_metric_values,
        ) = tt.predict_and_compute_metrics()
        metric_values[(task_type, dataset_name)]["recognition"][
            method_name
        ] = recognition_metric_values
        metric_values[(task_type, dataset_name)]["uncertainty"][
            method_name
        ] = uncertainty_metric_values

    # open set identif metric table

    if "open_set_identification_methods" in cfg:
        open_set_identification_result_dir = (
            Path(cfg.exp_dir) / dataset_name / "open_set_identification"
        )
        open_set_identification_result_dir.mkdir(exist_ok=True, parents=True)
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
