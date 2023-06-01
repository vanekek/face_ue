#!/usr/bin/env python3
import os
import numpy as np
from pathlib import Path
import hydra
from hydra.utils import instantiate
import sys

from evaluation.ijb_test import IJB_test
from evaluation.visualize import plot_dir_far_cmc_scores

path = str(Path(__file__).parent.parent.absolute())
sys.path.insert(1, path)


@hydra.main(
    config_path=str(Path(__file__).resolve().parents[1] / "configs/uncertainty_benchmark"),
    config_name=Path(__file__).stem,
    version_base="1.2",
)
def main(cfg):
    method_scores, method_names = [], []
    for method in cfg.open_set_recognition_methods:
        one_to_N_eval_function = instantiate(method.evaluation_1N_function)
        
        if hasattr(one_to_N_eval_function, '__name__'):
            save_name = one_to_N_eval_function.__name__
        else:
            save_name = os.path.splitext(os.path.basename(method.save_result))[0]
            
        save_path = os.path.dirname(method.save_result)
        save_items = {}
        if len(save_path) != 0 and not os.path.exists(save_path):
            os.makedirs(save_path)

        template_pooling = instantiate(method.template_pooling_strategy)
        tt = IJB_test(
            model_file=None,
            data_path=cfg.data_path,
            subset=cfg.subset,
            evaluation_1N_function=one_to_N_eval_function,
            batch_size=cfg.batch_size,
            force_reload=False,
            restore_embs=method.restore_embs,
            template_pooling_strategy=template_pooling,
            use_detector_score=method.use_detector_score,
            use_two_galleries=cfg.use_two_galleries,
            recompute_template_pooling=cfg.recompute_template_pooling,
            features=method.features,
            far_range=cfg.far_range,
        )

        if cfg.is_one_2_N:  # 1:N test
            fars, tpirs, _, _ = tt.run_model_test_1N()
            scores = [(fars, tpirs)]
            names = [save_name]
            method_scores.append((fars, tpirs))
            method_names.append(save_name)
            save_items.update({"scores": scores, "names": names})
        elif cfg.is_bunch:  # All 8 tests N{0,1}D{0,1}F{0,1}
            scores, names = tt.run_model_test_bunch()
            names = [save_name + "_" + ii for ii in names]
            label = tt.label
            save_items.update({"scores": scores, "names": names})
        else:  # Basic 1:1 N0D1F1 test
            score = tt.run_model_test_single()
            scores, names, label = [score], [save_name], tt.label
            save_items.update({"scores": scores, "names": names})

        np.savez(os.path.join(save_path, save_name + '.npz'), **save_items)

    fig = plot_dir_far_cmc_scores(scores=method_scores, names=method_names)
    fig.savefig(Path(cfg.exp_dir) / "di_far_plot.png", dpi=300)
    print("Plot path:")
    print(str(Path(cfg.exp_dir) / "di_far_plot.png"))
    # else:
    #     plot_roc_and_calculate_tpr(scores, names=names, label=label)


if __name__ == "__main__":
    main()
