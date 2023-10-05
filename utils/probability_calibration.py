import warnings

warnings.filterwarnings("ignore")

from pathlib import Path
import hydra
from hydra.utils import instantiate
from evaluation.face_recognition_test import Face_Fecognition_test
from evaluation.ijb_evals import instantiate_list, init_methods
from shutil import copyfile, rmtree
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from utils.reliability_diagrams import reliability_diagram


@hydra.main(
    config_path=str(Path(__file__).resolve().parents[1] / "configs/utils"),
    config_name=Path(__file__).stem,
    version_base="1.2",
)
def main(cfg):
    methods, method_task_type = init_methods(cfg)
    tasks_names = list(set(method_task_type))

    # instantiate datasets
    test_datasets = instantiate_list(cfg.test_datasets)
    dataset_names = [test_dataset.dataset_name for test_dataset in test_datasets]

    # instantiate method
    template_pooling = instantiate(methods[0].template_pooling_strategy)
    sampler = instantiate(methods[0].sampler)
    distance_function = instantiate(methods[0].distance_function)

    for test_dataset in test_datasets:
        dataset_name = test_dataset.dataset_name
        data_path = Path(test_dataset.dataset_path)
        embeddings_path = (
            Path(test_dataset.dataset_path)
            / f"embeddings/{methods[0].embeddings}_embs_{dataset_name}.npz"
        )
        tt = Face_Fecognition_test(
            task_type=tasks_names[0],
            method_name="test",
            recognition_method=None,
            sampler=sampler,
            distance_function=distance_function,
            test_dataset=test_dataset,
            embeddings_path=embeddings_path,
            template_pooling_strategy=template_pooling,
            use_detector_score=methods[0].use_detector_score,
            use_two_galleries=cfg.use_two_galleries,
            recompute_template_pooling=cfg.recompute_template_pooling,
            recognition_metrics=None,
            uncertainty_metrics=None,
        )

        used_galleries = ["g1"]
        if cfg.use_two_galleries:
            used_galleries += ["g2"]
        galleries_data = [
            tt.get_template_subsets(
                getattr(tt.test_dataset, f"{g}_templates"),
                getattr(tt.test_dataset, f"{g}_ids"),
            )
            for g in used_galleries
        ]
        (
            probe_templates_feature,
            probe_template_unc,
            probe_unique_ids,
        ) = tt.get_template_subsets(
            tt.test_dataset.probe_templates, tt.test_dataset.probe_ids
        )
        probe_unique_templates = np.unique(
            tt.test_dataset.probe_templates, return_index=False
        )

        # sample probe feature vectors
        probe_templates_feature = tt.sampler(
            probe_templates_feature,
            probe_template_unc,
        )
        for gallery_name, (g_templates_feature, g_template_unc, g_unique_ids) in zip(
            used_galleries, galleries_data
        ):
            similarity = tt.distance_function(
                probe_templates_feature,
                probe_template_unc,
                g_templates_feature,
                g_template_unc,
            )
            for method in methods:
                recognition_method = instantiate(method.recognition_method)
                # setup osr method and predict

                class_log_probs = recognition_method.get_class_log_probs(similarity)
                gallery_ids_with_imposter_id = np.concatenate([g_unique_ids, [-1]])
                predict_id = gallery_ids_with_imposter_id[
                    np.argmax(class_log_probs, axis=-1)
                ]
                conf_id = np.exp(np.max(class_log_probs, axis=-1))

                true_id = np.zeros_like(predict_id)
                is_seen = np.isin(probe_unique_ids, g_unique_ids)
                true_id[is_seen] = probe_unique_ids[is_seen]
                true_id[~is_seen] = -1

                plt.style.use("seaborn")

                plt.rc("font", size=12)
                plt.rc("axes", labelsize=12)
                plt.rc("xtick", labelsize=12)
                plt.rc("ytick", labelsize=12)
                plt.rc("legend", fontsize=12)

                plt.rc("axes", titlesize=16)
                plt.rc("figure", titlesize=16)

                title = method.pretty_name

                fig = reliability_diagram(
                    true_id,
                    predict_id,
                    conf_id,
                    num_bins=cfg.num_bins,
                    draw_ece=True,
                    draw_bin_importance=cfg.draw_bin_importance,
                    draw_averages=cfg.draw_averages,
                    title=title,
                    figsize=(6, 6),
                    dpi=300,
                    return_fig=True,
                )
                out_file = (
                    Path(cfg.exp_dir) / f"{dataset_name}_{gallery_name}_{title}.png"
                )
                fig.savefig(out_file, dpi=300)
                plt.close(fig)
    print(cfg.exp_dir)


if __name__ == "__main__":
    main()
