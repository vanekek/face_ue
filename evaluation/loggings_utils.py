from pathlib import Path
import pandas as pd
from evaluation.visualize import (
    plot_dir_far_scores,
    plot_cmc_scores,
    plot_rejection_scores,
)


def create_open_set_ident_recognition_metric_table(
    recognition_result_dict: dict,
    pretty_names: dict,
) -> pd.DataFrame:
    column_names = ["pretty_name", "model_name"]
    data_rows = []
    for i, model_name in enumerate(recognition_result_dict.keys()):
        metrics = []
        for metric_key, metric_value in recognition_result_dict[model_name].items():
            if "final" in metric_key:
                if i == 0:
                    column_names.append(metric_key)
                metrics.append(metric_value)
        data_rows.append([pretty_names[model_name], model_name] + metrics)
    # recognition_result_dict
    df = pd.DataFrame(data_rows, columns=column_names)
    return df


def create_open_set_ident_plots(
    recognition_result_dict: dict, out_dir: Path, pretty_names: dict
):
    metric_names = []
    for _, metric in recognition_result_dict.items():
        for key in metric.keys():
            if "metric:recalls" in key or "error_count:false-rejection-count_" in key:
                metric_names.append(key)
        break
    value_name_to_pretty_name = {
        "recalls": "Detection & Identification Rate",
        "false-rejection-count": "False Rejection Count",
    }
    for metric_name in metric_names:
        rank = metric_name.split("_")[-2]
        rank_dir = out_dir / f"rank_{rank}"
        rank_dir.mkdir(exist_ok=True)
        model_names = []
        scores = []
        for model_name, metrics in recognition_result_dict.items():
            model_names.append(pretty_names[model_name])
            scores.append((metrics["fars"], metrics[metric_name]))
        display_value_name = metric_name.split("_")[-3].split(":")[1]
        fig = plot_dir_far_scores(
            scores=scores,
            names=model_names,
            y_label=f"Rank {rank} {value_name_to_pretty_name[display_value_name]}",
        )
        fig.savefig(rank_dir / f"{display_value_name}.png", dpi=300)


def create_closed_set_ident_plots(
    recognition_result_dict: dict, out_dir: Path, pretty_names: dict
):
    model_names = []
    scores = []
    for method_name, metric in recognition_result_dict.items():
        model_names.append(pretty_names[method_name])
        scores.append((metric["ranks"], metric["cmc"]))

    fig = plot_cmc_scores(
        scores=scores,
        names=model_names,
    )
    fig.savefig(out_dir / f"cmc_plot.png", dpi=300)


def create_rejection_plots(
    open_set_uncertainty_result_metrics: dict, out_dir: Path, pretty_names: dict
):
    metric_names = []
    for _, metric in open_set_uncertainty_result_metrics.items():
        for key in metric.keys():
            if "plot_reject" in key:
                metric_names.append(key)
        break

    # create unified plot of different rejection metrics for each rank
    rank_metric_to_unc_metrics = {}
    for metric_name in metric_names:
        rank = metric_name.split("_")[-3]
        metric = metric_name.split("_")[-4]
        if (rank, metric) in rank_metric_to_unc_metrics:
            rank_metric_to_unc_metrics[(rank, metric)].append(metric_name)
        else:
            rank_metric_to_unc_metrics[(rank, metric)] = [metric_name]

    for (rank, metric), rank_metric_names in rank_metric_to_unc_metrics.items():
        model_names = []
        scores = []
        rank_dir = out_dir / f"rank_{rank}"
        rank_dir.mkdir(exist_ok=True)
        for metric_name in rank_metric_names:
            pretty_unc_metric_name = metric_name.split("_")[-1]
            for model_name, metrics in open_set_uncertainty_result_metrics.items():
                model_names.append(
                    pretty_names[model_name] + ";  " + pretty_unc_metric_name
                )
                scores.append((metrics["fractions"], metrics[metric_name]))

        fig = plot_rejection_scores(
            scores=scores,
            names=model_names,
            y_label=f"Ранг {rank} {metric}",
        )
        fig.savefig(rank_dir / f"{metric}_rejection.png", dpi=300)


def create_open_set_ident_uncertainty_metric_table(
    uncertainty_result_dict: dict,
) -> pd.DataFrame:
    pass
