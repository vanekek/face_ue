import numpy as np
from pathlib import Path


from .interfaces import (
    keras_model_interf,
    Torch_model_interf,
    ONNX_model_interf,
    Mxnet_model_interf,
)
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

        print(">>>> Reload embeddings from:", embeddings_path)
        aa = np.load(embeddings_path)

        self.embs = aa["embs"]
        self.embs_f = []
        self.unc = aa["unc"]

        if self.test_dataset.face_scores is not None:
            self.test_dataset.face_scores = self.test_dataset.face_scores.astype(
                self.embs.dtype
            )
        self.evaluation_1N_function = evaluation_1N_function
        self.template_pooling_strategy = template_pooling_strategy

        self.use_detector_score = use_detector_score
        self.far_range = far_range

        # process embeddings
        self.image_input_feats = process_embeddings(
            self.embs,
            self.embs_f,
            use_flip_test=False,
            use_norm_score=False,
            use_detector_score=self.use_detector_score,
            face_scores=self.test_dataset.face_scores,
        )
        # pool templates

        self.pool_templates(cache_dir="/app/cache/template_cache")

        self.fars_cal = [
            10**ii
            for ii in np.arange(
                self.far_range[0], self.far_range[1], 4.0 / self.far_range[2]
            )
        ] + [
            1
        ]  # plot in range [10-4, 1]

    def pool_templates(self, cache_dir: str):
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        class_name = self.template_pooling_strategy.__class__.__name__
        pooled_templates_path = (
            cache_dir
            / f"template_pool_{class_name}_det_score_{str(self.use_detector_score)}_{self.test_dataset.dataset_name}.npz"
        )
        if pooled_templates_path.is_file() and self.recompute_template_pooling is False:
            pooled_data = np.load(pooled_templates_path)
            self.template_pooled_emb = pooled_data["template_pooled_emb"]
            self.template_pooled_unc = pooled_data["template_pooled_unc"]
            self.template_ids = pooled_data["template_ids"]
        else:
            (
                self.template_pooled_emb,
                self.template_pooled_unc,
                self.template_ids,
            ) = self.template_pooling_strategy(
                self.image_input_feats,
                self.unc,
                self.test_dataset.templates,
                self.test_dataset.medias,
            )
            np.savez(
                pooled_templates_path,
                template_pooled_emb=self.template_pooled_emb,
                template_pooled_unc=self.template_pooled_unc,
                template_ids=self.template_ids,
            )

    def run_model_test_verification(
        self,
    ):
        template_norm_feats, template_unc, unique_templates, _ = image2template_feature(
            self.image_input_feats,
            self.test_dataset.templates,
            self.test_dataset.medias,
        )
        score = verification_11(
            template_norm_feats,
            unique_templates,
            self.test_dataset.p1,
            self.test_dataset.p2,
        )
        return score

    def get_template_subsets(
        self, choose_templates: np.ndarray, choose_ids: np.ndarray
    ):
        unique_templates, indices = np.unique(choose_templates, return_index=True)
        unique_subjectids = choose_ids[indices]

        templates_feature = np.zeros(
            (len(unique_templates), self.template_pooled_emb.shape[1])
        )
        template_unc = np.zeros(
            (len(unique_templates), self.template_pooled_unc.shape[1])
        )

        for count_template, uqt in enumerate(unique_templates):
            (ind_t,) = np.where(self.template_ids == uqt)
            templates_feature[count_template] = self.template_pooled_emb[ind_t]
            template_unc[count_template] = self.template_pooled_unc[ind_t]
        return templates_feature, template_unc, unique_subjectids

    def run_model_test_openset_identification(self):
        # pool first gallery
        (
            g1_templates_feature,
            g1_template_unc,
            g1_unique_ids,
        ) = self.get_template_subsets(
            self.test_dataset.g1_templates, self.test_dataset.g1_ids
        )

        if self.use_two_galleries and self.test_dataset.g2_templates.shape != ():
            # pool second gallery
            (
                g2_templates_feature,
                g2_template_unc,
                g2_unique_ids,
            ) = self.get_template_subsets(
                self.test_dataset.g2_templates, self.test_dataset.g2_ids
            )
        # pool probes
        (
            probe_templates_feature,
            probe_template_unc,
            probe_unique_ids,
        ) = self.get_template_subsets(
            self.test_dataset.probe_templates, self.test_dataset.probe_ids
        )

        print("g1_templates_feature:", g1_templates_feature.shape)  # (1772, 512)

        if self.use_two_galleries and self.test_dataset.g2_templates.shape != ():
            print("g2_templates_feature:", g2_templates_feature.shape)  # (1759, 512)
        print("probe_templates_feature:", probe_templates_feature.shape)  # (19593, 512)
        print("probe_unique_ids:", probe_unique_ids.shape)  # (19593,)

        print(">>>> Gallery 1")
        (
            g1_top_1_count,
            g1_top_5_count,
            g1_top_10_count,
            g1_threshes,
            g1_recalls,
            g1_cmc_scores,
        ) = self.evaluation_1N_function(
            probe_templates_feature,
            probe_template_unc,
            g1_templates_feature,
            g1_template_unc,
            probe_unique_ids,
            g1_unique_ids,
            self.fars_cal,
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
                probe_templates_feature,
                probe_template_unc,
                g2_templates_feature,
                g2_template_unc,
                probe_unique_ids,
                g2_unique_ids,
                self.fars_cal,
            )
            print(">>>> Mean")
            query_num = probe_templates_feature.shape[0]
            top_1 = (g1_top_1_count + g2_top_1_count) / query_num
            top_5 = (g1_top_5_count + g2_top_5_count) / query_num
            top_10 = (g1_top_10_count + g2_top_10_count) / query_num
            print("[Mean] top1: %f, top5: %f, top10: %f" % (top_1, top_5, top_10))

            mean_tpirs = (np.array(g1_recalls) + np.array(g2_recalls)) / 2
        else:
            query_num = probe_templates_feature.shape[0] - 3000
            top_1 = g1_top_1_count / query_num
            top_5 = g1_top_5_count / query_num
            top_10 = g1_top_10_count / query_num
            print("[Mean] top1: %f, top5: %f, top10: %f" % (top_1, top_5, top_10))
            mean_tpirs = np.array(g1_recalls)
        return self.fars_cal, mean_tpirs, None, None  # g1_cmc_scores, g2_cmc_scores
