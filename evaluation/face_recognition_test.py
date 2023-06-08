import numpy as np
from pathlib import Path


from .data_tools import extract_meta_data, extract_gallery_prob_data
from .interfaces import keras_model_interf, Torch_model_interf, ONNX_model_interf, Mxnet_model_interf
from .embeddings import get_embeddings, process_embeddings
from .image2template import image2template_feature
from .metrics import verification_11
from .template_pooling_strategies import AbstractTemplatePooling
from .eval_functions.abc import Abstract1NEval
from .test_datasets import FaceRecogntioniDataset

class Face_Fecognition_test:
    def __init__(
        self,
        evaluation_1N_function: Abstract1NEval,
        test_dataset: FaceRecogntioniDataset,
        embeddings_path: str,
        template_pooling_strategy: AbstractTemplatePooling,
        use_detector_score,
        use_two_galleries,
        recompute_template_pooling,
        far_range,
    ):
        self.use_two_galleries = use_two_galleries
        self.test_dataset = test_dataset
        self.recompute_template_pooling = recompute_template_pooling
        
        # (
        #     templates,
        #     medias,
        #     p1,
        #     p2,
        #     label,
        #     _,
        #     _,
        #     face_scores,
        # ) = extract_meta_data(data_path, subset, force_reload=force_reload)

        print(">>>> Reload embeddings from:", embeddings_path)
        aa = np.load(embeddings_path)

        self.embs = aa["embs"]
        self.embs_f = []
        self.unc = aa["unc"]
            
        
        # self.templates, self.medias, self.p1, self.p2, self.label = (
        #     templates,
        #     medias,
        #     p1,
        #     p2,
        #     label,
        # )
        if self.test_dataset.face_scores is not None:
            self.test_dataset.face_scores = self.test_dataset.face_scores.astype(self.embs.dtype)
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
            face_scores=self.test_dataset.face_scores,
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
        # (
        #     g1_templates,
        #     g1_ids,
        #     g2_templates,
        #     g2_ids,
        #     probe_mixed_templates,
        #     probe_mixed_ids,
        # ) = extract_gallery_prob_data(
        #     self.data_path, self.subset, force_reload=self.force_reload
        # )


        img_input_feats = process_embeddings(
            self.embs,
            self.embs_f,
            use_flip_test=False,
            use_norm_score=False,
            use_detector_score=self.use_detector_score,
            face_scores=self.test_dataset.face_scores,
        )
        (
            g1_templates_feature,
            g1_template_unc,
            g1_unique_templates,
            g1_unique_ids,
        ) = self.template_pooling_strategy(
            img_input_feats,
            self.unc,
            self.test_dataset.templates,
            self.test_dataset.medias,
            self.test_dataset.g1_templates,
            self.test_dataset.g1_ids,
        )
        if self.use_two_galleries and self.test_dataset.g2_templates.shape != ():
            (
                g2_templates_feature,
                g2_template_unc,
                g2_unique_templates,
                g2_unique_ids,
            ) = self.template_pooling_strategy(
                img_input_feats,
                self.unc,
                self.test_dataset.templates,
                self.test_dataset.medias,
                self.test_dataset.g2_templates,
                self.test_dataset.g2_ids,
            )
        cache_dir = Path("/app/cache/template_cache")
        cache_dir.mkdir(parents=True, exist_ok=True)
        class_name = self.template_pooling_strategy.__class__.__name__
        probe_mixed_templates_feature_path = str(
            cache_dir
            / f"probe_aggr_{class_name}_{str(self.use_detector_score)}_{self.test_dataset.dataset_name}"
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
                self.test_dataset.templates,
                self.test_dataset.medias,
                self.test_dataset.probe_mixed_templates,
                self.test_dataset.probe_mixed_ids,
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

        if self.use_two_galleries and self.test_dataset.g2_templates.shape != ():
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

        if self.use_two_galleries and self.test_dataset.g2_templates.shape != ():
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
            query_num = probe_mixed_templates_feature.shape[0] - 3000
            top_1 = g1_top_1_count / query_num
            top_5 = g1_top_5_count / query_num
            top_10 = g1_top_10_count / query_num
            print("[Mean] top1: %f, top5: %f, top10: %f" % (top_1, top_5, top_10))
            mean_tpirs = np.array(g1_recalls)
        return fars_cal, mean_tpirs, None, None  # g1_cmc_scores, g2_cmc_scores