import numpy as np
from pathlib import Path
import importlib

from .data_tools import extract_IJB_data_11, extract_gallery_prob_data
from .interfaces import keras_model_interf, Torch_model_interf, ONNX_model_interf, Mxnet_model_interf
from .embeddings import get_embeddings, process_embeddings
from .image2template import image2template_feature
from .metrics import verification_11


class IJB_test:
    def __init__(
        self,
        model_file,
        data_path,
        subset,
        evaluation_1N_function,
        batch_size,
        force_reload,
        restore_embs,
        template_pooling_strategy,
        use_detector_score,
        use_two_galleries,
        recompute_template_pooling,
        features,
        far_range,
    ):
        self.use_two_galleries = use_two_galleries
        self.recompute_template_pooling = recompute_template_pooling
        self.features = features
        (
            templates,
            medias,
            p1,
            p2,
            label,
            img_names,
            landmarks,
            face_scores,
        ) = extract_IJB_data_11(data_path, subset, force_reload=force_reload)
        if model_file != None:
            if model_file.endswith(".h5"):
                interf_func = keras_model_interf(model_file)
            elif model_file.endswith(".pth") or model_file.endswith(".pt"):
                interf_func = Torch_model_interf(model_file)
            elif model_file.endswith(".onnx") or model_file.endswith(".ONNX"):
                interf_func = ONNX_model_interf(model_file)
            else:
                interf_func = Mxnet_model_interf(model_file)
            self.embs, self.embs_f = get_embeddings(
                interf_func, img_names, landmarks, batch_size=batch_size
            )
        elif restore_embs != None:
            print(">>>> Reload embeddings from:", restore_embs)
            aa = np.load(restore_embs)

            if "embs" in aa and "unc" in aa:
                self.embs = aa["embs"]
                self.embs_f = []
                self.unc = aa["unc"]
            else:
                print("ERROR: %s NOT containing embs / unc" % restore_embs)
                exit(1)
            print(">>>> Done.")
        self.data_path, self.subset, self.force_reload = data_path, subset, force_reload
        self.templates, self.medias, self.p1, self.p2, self.label = (
            templates,
            medias,
            p1,
            p2,
            label,
        )
        self.face_scores = face_scores.astype(self.embs.dtype)
        self.evaluation_1N_function = evaluation_1N_function
        self.template_pooling_strategy = template_pooling_strategy

        self.use_detector_score = use_detector_score
        self.far_range = far_range

    def run_model_test_single(
        self, use_flip_test=True, use_norm_score=False, use_detector_score=True
    ):
        img_input_feats = process_embeddings(
            self.embs,
            self.embs_f,
            use_flip_test=use_flip_test,
            use_norm_score=use_norm_score,
            use_detector_score=use_detector_score,
            face_scores=self.face_scores,
        )
        template_norm_feats, template_unc, unique_templates, _ = image2template_feature(
            img_input_feats, self.templates, self.medias
        )
        score = verification_11(template_norm_feats, unique_templates, self.p1, self.p2)
        return score

    def run_model_test_bunch(self):
        from itertools import product

        scores, names = [], []
        for use_norm_score, use_detector_score, use_flip_test in product(
            [True, False], [True, False], [True, False]
        ):
            name = "N{:d}D{:d}F{:d}".format(
                use_norm_score, use_detector_score, use_flip_test
            )
            print(">>>>", name, use_norm_score, use_detector_score, use_flip_test)
            names.append(name)
            scores.append(
                self.run_model_test_single(
                    use_flip_test, use_norm_score, use_detector_score
                )
            )
        return scores, names

    def run_model_test_1N(self, npoints=100):
        fars_cal = [
            10**ii
            for ii in np.arange(
                self.far_range[0], self.far_range[1], 4.0 / self.far_range[2]
            )
        ] + [
            1
        ]  # plot in range [10-4, 1]
        fars_show_idx = np.arange(len(fars_cal))[
            :: npoints // 4
        ]  # npoints=100, fars_show=[0.0001, 0.001, 0.01, 0.1, 1.0]

        (
            g1_templates,
            g1_ids,
            g2_templates,
            g2_ids,
            probe_mixed_templates,
            probe_mixed_ids,
        ) = extract_gallery_prob_data(
            self.data_path, self.subset, force_reload=self.force_reload
        )
        img_input_feats = process_embeddings(
            self.embs,
            self.embs_f,
            use_flip_test=False,
            use_norm_score=False,
            use_detector_score=self.use_detector_score,
            face_scores=self.face_scores,
        )
        # get template pooling function
        (
            g1_templates_feature,
            g1_template_unc,
            g1_unique_templates,
            g1_unique_ids,
        ) = self.template_pooling_strategy(
            img_input_feats,
            self.unc,
            self.templates,
            self.medias,
            g1_templates,
            g1_ids,
        )
        if self.use_two_galleries:
            (
                g2_templates_feature,
                g2_template_unc,
                g2_unique_templates,
                g2_unique_ids,
            ) = self.template_pooling_strategy(
                img_input_feats,
                self.unc,
                self.templates,
                self.medias,
                g2_templates,
                g2_ids,
            )
        cache_dir = Path("/app/cache/template_cache")
        cache_dir.mkdir(parents=True, exist_ok=True)
        class_name = self.template_pooling_strategy.__class__.__name__
        probe_mixed_templates_feature_path = str(
            cache_dir
            / f"probe_aggr_{class_name}_{str(self.use_detector_score)}_{self.subset}"
        )

        if (
            Path(probe_mixed_templates_feature_path + "_unc.npy").is_file()
            and self.recompute_template_pooling is False
        ):
            probe_mixed_templates_feature = np.load(
                probe_mixed_templates_feature_path + "_feature.npy"
            )
            probe_template_unc = np.load(
                probe_mixed_templates_feature_path + "_unc.npy"
            )
            probe_mixed_unique_subject_ids = np.load(
                probe_mixed_templates_feature_path + "_subject_ids.npy"
            )
        else:
            (
                probe_mixed_templates_feature,
                probe_template_unc,
                probe_mixed_unique_templates,
                probe_mixed_unique_subject_ids,
            ) = self.template_pooling_strategy(
                img_input_feats,
                self.unc,
                self.templates,
                self.medias,
                probe_mixed_templates,
                probe_mixed_ids,
            )
            np.save(
                probe_mixed_templates_feature_path + "_feature.npy",
                probe_mixed_templates_feature,
            )
            np.save(
                probe_mixed_templates_feature_path + "_unc.npy",
                probe_template_unc,
            )
            np.save(
                probe_mixed_templates_feature_path + "_subject_ids.npy",
                probe_mixed_unique_subject_ids,
            )
        print("g1_templates_feature:", g1_templates_feature.shape)  # (1772, 512)

        if self.use_two_galleries:
            print("g2_templates_feature:", g2_templates_feature.shape)  # (1759, 512)

        print(
            "probe_mixed_templates_feature:", probe_mixed_templates_feature.shape
        )  # (19593, 512)
        print(
            "probe_mixed_unique_subject_ids:", probe_mixed_unique_subject_ids.shape
        )  # (19593,)

        print(">>>> Gallery 1")
        (
            g1_top_1_count,
            g1_top_5_count,
            g1_top_10_count,
            g1_threshes,
            g1_recalls,
            g1_cmc_scores,
        ) = self.evaluation_1N_function(
            probe_mixed_templates_feature,
            probe_template_unc,
            g1_templates_feature,
            g1_template_unc,
            probe_mixed_unique_subject_ids,
            g1_unique_ids,
            fars_cal,
        )

        if self.use_two_galleries:
            print(">>>> Gallery 2")
            (
                g2_top_1_count,
                g2_top_5_count,
                g2_top_10_count,
                g2_threshes,
                g2_recalls,
                g2_cmc_scores,
            ) = self.evaluation_1N_function(
                probe_mixed_templates_feature,
                probe_template_unc,
                g2_templates_feature,
                g2_template_unc,
                probe_mixed_unique_subject_ids,
                g2_unique_ids,
                fars_cal,
            )
            print(">>>> Mean")
            query_num = probe_mixed_templates_feature.shape[0]
            top_1 = (g1_top_1_count + g2_top_1_count) / query_num
            top_5 = (g1_top_5_count + g2_top_5_count) / query_num
            top_10 = (g1_top_10_count + g2_top_10_count) / query_num
            print("[Mean] top1: %f, top5: %f, top10: %f" % (top_1, top_5, top_10))

            mean_tpirs = (np.array(g1_recalls) + np.array(g2_recalls)) / 2
        else:
            mean_tpirs = np.array(g1_recalls)
        return fars_cal, mean_tpirs, None, None  # g1_cmc_scores, g2_cmc_scores