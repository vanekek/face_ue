import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
import hydra


def compute_best_values_for_each_horizon(ts_metric_values, rank_values):
    best_mae = np.sort(ts_metric_values["MAE"].values)
    best_rmse = np.sort(ts_metric_values["RMSE"].values)
    if rank_values:
        best_r2 = np.sort(ts_metric_values["R2"].values)
    else:
        best_r2 = np.sort(ts_metric_values["R2"].values)[::-1]

    best_rmsce = np.sort(ts_metric_values["Root-mean-squared Calibration Error"].values)
    best_mical = np.sort(ts_metric_values["Miscalibration Area"].values)
    if rank_values:
        best_picp = np.sort(ts_metric_values["picp"].values)
    else:
        best_picp = np.sort(ts_metric_values["picp"].values)[::-1]
    best_ence = np.sort(ts_metric_values["ence"].values)
    if rank_values:
        best_std_variation = np.sort(ts_metric_values["std_variation"].values)
    else:
        best_std_variation = np.sort(ts_metric_values["std_variation"].values)[::-1]
    best_aggregated_ence_cv = np.sort(ts_metric_values["aggregated_ence_cv"].values)

    best_values = {
        "MAE": (best_mae[0], best_mae[1]),
        "RMSE": (best_rmse[0], best_rmse[1]),
        "R2": (best_r2[0], best_r2[1]),
        "Root-mean-squared Calibration Error": (best_rmsce[0], best_rmsce[1]),
        "Miscalibration Area": (best_mical[0], best_mical[1]),
        "picp": (best_picp[0], best_picp[1]),
        "ence": (best_ence[0], best_ence[1]),
        "std_variation": (best_std_variation[0], best_std_variation[1]),
        "aggregated_ence_cv": (best_aggregated_ence_cv[0], best_aggregated_ence_cv[1]),
    }
    return best_values


def create_table_head(result_latex_code, caption, table_lable, cfg):
    used_columns = cfg.used_columns
    column_pretty_name = cfg.pretty_name.column

    result_latex_code += "\\begin{table}\n"
    if cfg.use_scriptsize:
        result_latex_code += "\\scriptsize\n"
    result_latex_code += "\\caption{" + caption + "}\n"
    result_latex_code += "\\label{" + table_lable + "}\n"
    if cfg.use_adjustbox:
        result_latex_code += "\\begin{adjustbox}{width=0.5\\textwidth}\n"
    result_latex_code += "\\begin{tabular}{" + "c" * len(used_columns) + "}\n"
    result_latex_code += "\\toprule\n"

    next_column_index = 1
    for raw_column_name in used_columns:
        pretty_name = column_pretty_name[raw_column_name]
        if isinstance(pretty_name, str):
            pretty_name = [pretty_name]
        result_latex_code += (
            "\\begin{tabular}{c}\n" + "\\\\\n".join(pretty_name) + "\n\\end{tabular}"
        )
        if next_column_index < len(used_columns):
            result_latex_code += "&\n"
        else:
            result_latex_code += "\\\\\n"
            result_latex_code += "\midrule\n"
        next_column_index += 1
    return result_latex_code


def draw_column(data, pretty_name_model, column_name, table_lable):
    for dataset_name, dataset_metric_values in data.groupby(
        ["dataset name"], sort=True
    ):
        plt.figure(figsize=(12, 6))
        for model_name, model_metrics in dataset_metric_values.groupby(["model"]):
            plt.plot(
                [
                    int(ts_name.split(" ")[1])
                    for ts_name in model_metrics["ts in-ds name"]
                ],
                model_metrics[column_name],
                label=f"{pretty_name_model[model_name]}",
                linewidth=2,
                markersize=4,
            )
        # plt.xticks(fontsize=14, rotation=60)
        plt.yticks(fontsize=14)
        plt.xlabel(f"Horizon", fontsize=18)
        plt.ylabel(f"{column_name}", fontsize=18)
        plt.legend(fontsize=18)
        plt.savefig(f"{dataset_name}_{table_lable}.pdf")


def create_table_body(result_latex_code, cfg):
    all_metric_values = pd.read_csv(
        Path(hydra.utils.get_original_cwd()) / cfg.metric_table_path
    )
    # add anxilary columns
    if "dataset name" not in all_metric_values.columns:
        all_metric_values["dataset name"] = list(
            len(all_metric_values) * ["all_fd_and_datasets_A_B"]
        )
        all_metric_values["ts in-ds name"] = list(len(all_metric_values) * ["none"])

    selected_metric_values = all_metric_values[
        (all_metric_values["dataset name"].isin(cfg.used_datasets))
        & (all_metric_values["model"].isin(cfg.used_models))
        & (all_metric_values["ts in-ds name"].isin(cfg.used_ts_names))
    ]
    # draw table
    if hasattr(cfg, "draw_column"):
        draw_column(
            selected_metric_values,
            cfg.pretty_name.model,
            cfg.draw_column,
            cfg.table_lable.split(":")[-1],
        )
    next_ds_index = 1
    for dataset_name, dataset_metric_values in selected_metric_values.groupby(
        ["dataset name"], sort=True
    ):
        if "dataset name" in cfg.used_columns:
            num_dataset_rows = len(dataset_metric_values)
            result_latex_code += (
                "\\multirow{"
                + str(num_dataset_rows)
                + "}{*}{"
                + cfg.pretty_name.dataset[dataset_name]
                + "} & "
            )
        indataset_row = 0
        next_ts_index = 1
        for ts_name, ts_metric_values in dataset_metric_values.groupby(
            ["ts in-ds name"], sort=False
        ):
            best_values = compute_best_values_for_each_horizon(
                ts_metric_values, cfg.rank_values
            )
            ts_metric_values = ts_metric_values.set_index("model")
            if cfg.sort_models_with == "ts":
                ts_metric_values = ts_metric_values.reindex(
                    cfg.ts_to_model_order[ts_name.split(" ")[1]]
                )
            else:
                ts_metric_values = ts_metric_values.reindex(cfg.used_models)

            for row_index, (model, row) in enumerate(ts_metric_values.iterrows()):
                if "dataset name" in cfg.used_columns and indataset_row != 0:
                    result_latex_code += "& "
                indataset_row += 1
                for column_index, column_name in enumerate(cfg.used_columns):
                    if column_name == "model":
                        result_latex_code += cfg.pretty_name.model[model] + " & "
                    elif column_name == "ts in-ds name":
                        if row_index == 0:
                            num_rows = (
                                len(cfg.ts_to_model_order[ts_name.split(" ")[1]])
                                if cfg.sort_models_with == "ts"
                                else len(cfg.used_models)
                            )
                            result_latex_code += (
                                "\\multirow{"
                                + str(num_rows)
                                + "}{*}{"
                                + ts_name.split(" ")[1]
                                + "} & "
                            )
                        else:
                            result_latex_code += "& "
                    elif column_name != "dataset name":
                        # metric value
                        metric_value = row[column_name]
                        if metric_value == best_values[column_name][0]:
                            # best value
                            result_latex_code += (
                                "\\textbf{" + str(np.round(metric_value, 3)) + "} "
                            )
                        elif metric_value == best_values[column_name][1]:
                            # second best value
                            result_latex_code += (
                                "\\underline{" + str(np.round(metric_value, 3)) + "} "
                            )
                        else:
                            result_latex_code += f" {str(np.round(metric_value, 3))} "
                        if column_index < len(cfg.used_columns) - 1:
                            result_latex_code += "& "
                        else:
                            # end of row
                            result_latex_code += "\\\\\n"

            # add hline at the end of ts
            if next_ts_index < len(cfg.used_ts_names):

                result_latex_code += "\\hline\n"
            next_ts_index += 1

        # add extra hline at the end of ds
        if next_ds_index < len(cfg.used_datasets):
            result_latex_code += "\\hline\n\\hline\n"
        next_ds_index += 1
    return result_latex_code


def create_table_tail(result_latex_code, cfg):
    result_latex_code += "\\bottomrule\n"
    result_latex_code += "\\end{tabular}\n"
    if cfg.use_adjustbox:
        result_latex_code += "\\end{adjustbox}\n"
    result_latex_code += "\\end{table}\n"

    return result_latex_code


@hydra.main(config_path="configs", config_name=Path(__file__).stem)
def run(cfg):
    result_latex_code = """"""
    dataset_names = [
        getattr(cfg.pretty_name.dataset, dataset_name)
        for dataset_name in cfg.used_datasets
    ]
    if "{dataset_name}" in cfg.caption:
        caption = cfg.caption.format(dataset_name=" and ".join(dataset_names))
    else:
        caption = cfg.caption

    if "{dataset_name}" in cfg.table_lable:
        table_lable = cfg.table_lable.format(dataset_name=cfg.used_datasets[0])
    else:
        table_lable = cfg.table_lable

    result_latex_code = create_table_head(
        result_latex_code,
        caption,
        table_lable,
        cfg,
    )

    result_latex_code = create_table_body(result_latex_code, cfg)

    result_latex_code = create_table_tail(result_latex_code, cfg)

    # save result
    with open("table.tex", "w") as fd:
        fd.write(result_latex_code)


if __name__ == "__main__":
    run()
