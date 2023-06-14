import numpy as np
from pathlib import Path


from .embeddings import process_embeddings
from .image2template import image2template_feature
from .template_pooling_strategies import AbstractTemplatePooling
from .eval_functions.open_set_identification.abc import Abstract1NEval
from .test_datasets import FaceRecogntioniDataset


class Face_Fecognition_test:
    def __init__(
        self,
        evaluation_function: Abstract1NEval,
        test_dataset: FaceRecogntioniDataset,
        embeddings_path: str,
        template_pooling_strategy: AbstractTemplatePooling,
        use_detector_score,
        use_two_galleries,
        recompute_template_pooling,
        open_set_identification_metrics,
        closed_set_identification_metrics,
        verification_metrics,
        varif_far_range,
        open_set_ident_far_range,
    ):
        self.use_two_galleries = use_two_galleries
        self.test_dataset = test_dataset
        self.recompute_template_pooling = recompute_template_pooling
        self.open_set_identification_metrics = open_set_identification_metrics
        self.closed_set_identification_metrics = closed_set_identification_metrics
        self.verification_metrics = verification_metrics

        print(">>>> Reload embeddings from:", embeddings_path)
        aa = np.load(embeddings_path)

        self.embs = aa["embs"]
        self.embs_f = []
        self.unc = aa["unc"]

        if self.test_dataset.face_scores is not None:
            self.test_dataset.face_scores = self.test_dataset.face_scores.astype(
                self.embs.dtype
            )
        self.evaluation_function = evaluation_function
        self.template_pooling_strategy = template_pooling_strategy

        self.use_detector_score = use_detector_score

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
        self.verif_far = [
            10**ii
            for ii in np.arange(
                varif_far_range[0], varif_far_range[1], 4.0 / varif_far_range[2]
            )
        ] + [1]
        self.open_set_ident_far = [
            10**ii
            for ii in np.arange(
                open_set_ident_far_range[0],
                open_set_ident_far_range[1],
                4.0 / open_set_ident_far_range[2],
            )
        ] + [1]

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

    def run_model_test_verification(
        self,
    ):
        scores = self.evaluation_function(
            self.template_pooled_emb,
            self.template_pooled_unc,
            self.template_ids[:20003],
            self.test_dataset.p1[:20003],
            self.test_dataset.p2[:20003],
        )

        metrics = {}
        for metric in self.verification_metrics:
            metrics.update(
                metric(
                    fars=self.verif_far,
                    scores=scores,
                    labels=self.test_dataset.label[:20003],
                )
            )
        return self.verif_far, metrics
    def run_model_test_closedset_identification(self):
        pass
    def run_model_test_openset_identification(self):
        (
            g1_templates_feature,
            g1_template_unc,
            g1_unique_ids,
        ) = self.get_template_subsets(
            self.test_dataset.g1_templates, self.test_dataset.g1_ids
        )

        if self.use_two_galleries and self.test_dataset.g2_templates.shape != ():
            (
                g2_templates_feature,
                g2_template_unc,
                g2_unique_ids,
            ) = self.get_template_subsets(
                self.test_dataset.g2_templates, self.test_dataset.g2_ids
            )
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
        labels_sorted = (
            True if self.evaluation_function.__class__.__name__ == "SVM" else False
        )
        similarity, probe_score = self.evaluation_function(
            probe_templates_feature,
            probe_template_unc,
            g1_templates_feature,
            g1_template_unc,
            probe_unique_ids,
            g1_unique_ids,
            self.open_set_ident_far,
        )
        metrics = {}
        for metric in self.open_set_identification_metrics:
            metrics.update(
                metric(
                    fars=self.open_set_ident_far,
                    probe_ids=probe_unique_ids,
                    gallery_ids=g1_unique_ids,
                    similarity=similarity,
                    probe_score=probe_score,
                    labels_sorted=labels_sorted,
                )
            )

        if self.use_two_galleries and self.test_dataset.g2_templates.shape != ():
            print(">>>> Gallery 2")
            similarity, probe_score = self.evaluation_function(
                probe_templates_feature,
                probe_template_unc,
                g2_templates_feature,
                g2_template_unc,
                probe_unique_ids,
                g2_unique_ids,
                self.open_set_ident_far,
            )
            g2_metrics = {}
            for metric in self.open_set_identification_metrics:
                g2_metrics.update(
                    metric(
                        fars=self.open_set_ident_far,
                        probe_ids=probe_unique_ids,
                        gallery_ids=g2_unique_ids,
                        similarity=similarity,
                        probe_score=probe_score,
                        labels_sorted=labels_sorted,
                    )
                )
            query_num = probe_templates_feature.shape[0]
            for key in g2_metrics.keys():
                if key == "recalls":
                    metrics[key] = (metrics[key] + g2_metrics[key]) / 2
                elif "top" in key:
                    metrics[key] = (metrics[key] + g2_metrics[key]) / query_num
                else:
                    raise ValueError
            print(">>>> Mean")

        else:
            is_seen = np.isin(probe_unique_ids, g1_unique_ids)
            query_num = len(is_seen)
            for key in metrics.keys():
                if key == "recalls":
                    pass
                elif "top" in key:
                    metrics[key] = (metrics[key]) / query_num
                else:
                    raise ValueError
        return self.open_set_ident_far, metrics
