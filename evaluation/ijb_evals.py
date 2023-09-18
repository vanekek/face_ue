#!/usr/bin/env python3

from pathlib import Path
import hydra
from omegaconf import OmegaConf
from hydra.utils import instantiate
import numpy as np
from itertools import product
from evaluation.face_recognition_test import Face_Fecognition_test
import pandas as pd
import matplotlib.pyplot as plt
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


def create_method_name(
    method, sampler, template_pooling, distance_function, recognition_method
):
    method_name_parts = []
    method_name_parts.append(
        f"sampler-{sampler.__class__.__name__}-num-samples-{sampler.num_samples}"
    )
    method_name_parts.append(f"pooling-with-{template_pooling.__class__.__name__}")
    method_name_parts.append(f"use-det-score-{method.use_detector_score}")
    method_name_parts.append(f"distance-{distance_function.__class__.__name__}")
    method_name_parts.append(f"osr-method-{recognition_method.__class__.__name__}")
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
    test_datasets = instantiate_list(cfg.test_datasets)
    dataset_names = [test_dataset.dataset_name for test_dataset in test_datasets]
    # create result dictionary
    metric_values = {
        (task, dataset_name): {"recognition": {}, "uncertainty": {}}
        for task, dataset_name in product(tasks_names, dataset_names)
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
        method_name = (
            create_method_name(
                method, sampler, template_pooling, distance_function, recognition_method
            )
            + f"_{method.pretty_name}"
        )
        print(method_name)
        pretty_names[task_type][method_name] = method.pretty_name
        embeddings_path = (
            Path(test_dataset.dataset_path)
            / f"embeddings/{method.embeddings}_embs_{dataset_name}.npz"
        )
        # create tester
        tt = Face_Fecognition_test(
            task_type=task_type,
            method_name=method_name,
            recognition_method=recognition_method,
            sampler=sampler,
            distance_function=distance_function,
            test_dataset=test_dataset,
            embeddings_path=embeddings_path,
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

    # create plots and tables
    # load metric name converter
    metric_pretty_name = OmegaConf.load(cfg.metric_pretty_name_path)
    for task_type, dataset_name in metric_values:
        # create output dir
        out_dir = Path(cfg.exp_dir) / str(dataset_name) / str(task_type)
        out_table_dir = out_dir / "tabels"
        out_table_fractions_dir = out_table_dir / "fractions"
        out_table_fractions_dir.mkdir(exist_ok=True, parents=True)
        # create rejection plots
        metric_names = []
        model_names = []

        for model_name, metric in metric_values[(task_type, dataset_name)][
            "uncertainty"
        ].items():
            for key in metric:
                if "osr_unc_metric" in key:
                    metric_names.append(key)
                    model_names.append(model_name)
            break
        fractions = next(
            iter(metric_values[(task_type, dataset_name)]["uncertainty"].items())
        )[1]["fractions"]
        fraction_data_rows = {frac: [] for frac in fractions}
        fraction_column_names = ["models"] + [
            metric_name.split(":")[-1] for metric_name in metric_names
        ]
        column_names = ["models", *[str(np.round(frac, 4)) for frac in fractions]]

        for method_name, metrics in metric_values[(task_type, dataset_name)][
            "uncertainty"
        ].items():
            for i, frac in enumerate(fractions):
                frac_data_rows = [pretty_names[task_type][method_name]]
                for metric_name in metric_names:
                    frac_data_rows.append(metrics[metric_name][i])
                fraction_data_rows[frac].append(frac_data_rows)
        for metric_name in metric_names:
            model_names = []
            scores = []
            data_rows = []
            for method_name, metrics in metric_values[(task_type, dataset_name)][
                "uncertainty"
            ].items():
                model_names.append(pretty_names[task_type][method_name])
                scores.append((metrics["fractions"], metrics[metric_name]))
                data_rows.append(
                    [pretty_names[task_type][method_name], *metrics[metric_name]]
                )
            pretty_name = metric_pretty_name[metric_name.split(":")[-1]]
            if isinstance(pretty_name, str):
                pretty_name = [pretty_name]
            pretty_name = " ".join(pretty_name)
            fig = plot_rejection_scores(
                scores=scores,
                names=model_names,
                y_label=f"{pretty_name}",
            )
            fig.savefig(
                out_dir / f"{metric_name.split(':')[-1]}_rejection.png", dpi=300
            )
            plt.close(fig)

            # save table

            rejection_df = pd.DataFrame(data_rows, columns=column_names)
            rejection_df.to_csv(
                out_table_dir / f'{metric_name.split(":")[-1]}_rejection.csv'
            )
        for frac, data_rows in fraction_data_rows.items():
            frac_rejection_df = pd.DataFrame(data_rows, columns=fraction_column_names)
            frac_rejection_df.to_csv(
                out_table_fractions_dir
                / f'{str(np.round(frac, 4)).ljust(6, "0")}_frac_rejection.csv'
            )
    print(cfg.exp_dir)


if __name__ == "__main__":
    main()
