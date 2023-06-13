#!/usr/bin/env python3
import os
import numpy as np
from pathlib import Path
import hydra
from hydra.utils import instantiate
import sys

from evaluation.face_recognition_test import Face_Fecognition_test
from evaluation.visualize import plot_dir_far_cmc_scores

path = str(Path(__file__).parent.parent.absolute())
sys.path.insert(1, path)


@hydra.main(
    config_path=str(
        Path(__file__).resolve().parents[1] / "configs/uncertainty_benchmark"
    ),
    config_name=Path(__file__).stem,
    version_base="1.2",
)
def main(cfg):
    test_dataset = instantiate(cfg.test_dataset)

    method_scores, method_names = [], []
    for method in cfg.open_set_recognition_methods:
        one_to_N_eval_function = instantiate(method.evaluation_1N_function)

        if hasattr(one_to_N_eval_function, "__name__"):
            save_name = one_to_N_eval_function.__name__
        else:
            save_name = os.path.splitext(os.path.basename(method.save_result))[0]

        template_pooling = instantiate(method.template_pooling_strategy)
        tt = Face_Fecognition_test(
            evaluation_1N_function=one_to_N_eval_function,
            test_dataset=test_dataset,
            embeddings_path=method.embeddings_path,
            template_pooling_strategy=template_pooling,
            use_detector_score=method.use_detector_score,
            use_two_galleries=cfg.use_two_galleries,
            recompute_template_pooling=cfg.recompute_template_pooling,
            far_range=cfg.far_range,
        )

        save_path = os.path.dirname(method.save_result)
        save_items = {}
        if len(save_path) != 0 and not os.path.exists(save_path):
            os.makedirs(save_path)
        if cfg.task == "openset_identification":  # 1:N test
            fars, tpirs, _, _ = tt.run_model_test_openset_identification()
            scores = [(fars, tpirs)]
            names = [save_name]
            method_scores.append((fars, tpirs))
            method_names.append(save_name)
            save_items.update({"scores": scores, "names": names})
        elif cfg.task == "verification":  # Basic 1:1 N0D1F1 test
            score = tt.run_model_test_verification()
            scores, names, label = [score], [save_name], tt.label
            save_items.update({"scores": scores, "names": names, "label": tt.label})
        elif cfg.task == "closedset_identification":
            pass
        else:
            raise ValueError
        np.savez(os.path.join(save_path, save_name + ".npz"), **save_items)

    fig = plot_dir_far_cmc_scores(scores=method_scores, names=method_names)
    fig.savefig(Path(cfg.exp_dir) / "di_far_plot.png", dpi=300)
    print("Plot path:")
    print(str(Path(cfg.exp_dir) / "di_far_plot.png"))
    # else:
    #     plot_roc_and_calculate_tpr(scores, names=names, label=label)


if __name__ == "__main__":
    main()
