import numpy as np
from pathlib import Path
import warnings

from .embeddings import process_embeddings
from .image2template import image2template_feature
from .template_pooling_strategies import AbstractTemplatePooling
from .distance_functions.open_set_identification.abc import Abstract1NEval
from .test_datasets import FaceRecogntioniDataset


class Face_Fecognition_test:
    def __init__(
        self,
        task_type: str,
        method_name: str,
        recognition_method,
        sampler,
        distance_function: Abstract1NEval,
        test_dataset: FaceRecogntioniDataset,
        embeddings_path: str,
        template_pooling_strategy: AbstractTemplatePooling,
        use_detector_score: bool,
        use_two_galleries: bool,
        recompute_template_pooling: bool,
        recognition_metrics: dict,
        uncertainty_metrics: dict,
    ):
        self.task_type = task_type
        self.method_name = method_name
        self.recognition_method = recognition_method
        self.use_two_galleries = use_two_galleries
        self.test_dataset = test_dataset
        self.recompute_template_pooling = recompute_template_pooling
        self.recognition_metrics = recognition_metrics
        self.uncertainty_metrics = uncertainty_metrics
        self.sampler = sampler
        self.distance_function = distance_function
        self.template_pooling_strategy = template_pooling_strategy
        self.use_detector_score = use_detector_score

        # load nn embeddings
        aa = np.load(embeddings_path)
        self.embeddings_path = embeddings_path
        self.embs = aa["embs"]
        self.embs_f = []
        self.unc = aa["unc"]
        if self.test_dataset.face_scores is not None:
            self.test_dataset.face_scores = self.test_dataset.face_scores.astype(
                self.embs.dtype
            )

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

    def predict_and_compute_metrics(self):
        return getattr(self, f"run_model_test_{self.task_type}")()

    def run_model_test_open_set_identification(self):
        used_galleries = ["g1"]
        if self.use_two_galleries and self.test_dataset.g2_templates.shape != ():
            used_galleries += ["g2"]

        galleries_data = [
            self.get_template_subsets(
                getattr(self.test_dataset, f"{g}_templates"),
                getattr(self.test_dataset, f"{g}_ids"),
            )
            for g in used_galleries
        ]
        (
            probe_templates_feature,
            probe_template_unc,
            probe_unique_ids,
        ) = self.get_template_subsets(
            self.test_dataset.probe_templates, self.test_dataset.probe_ids
        )

        # sample probe feature vectors
        probe_templates_feature = self.sampler(
            probe_templates_feature,
            probe_template_unc,
        )
        metrics = {gallery: {} for gallery in used_galleries}
        unc_metrics = {gallery: {} for gallery in used_galleries}
        for gallery_name, (g_templates_feature, g_template_unc, g_unique_ids) in zip(
            used_galleries, galleries_data
        ):
            similarity = self.distance_function(
                probe_templates_feature,
                probe_template_unc,
                g_templates_feature,
                g_template_unc,
            )

            # setup osr method and predict
            self.recognition_method.setup(similarity)
            predicted_id, was_rejected = self.recognition_method.predict()
            predicted_unc = self.recognition_method.predict_uncertainty(
                probe_template_unc
            )

            # compute recognition metrics
            # sort labels
            # gallery_ids_argsort = np.argsort(g_unique_ids)
            # g_unique_ids_sorted = g_unique_ids[gallery_ids_argsort]
            # g_unique_ids_sorted = np.concatenate(
            #     [g_unique_ids, [-1]]
            # )  # last id is for out of gallery class

            for metric in self.recognition_metrics[self.task_type]:
                metrics[gallery_name].update(
                    metric(
                        predicted_id=predicted_id,
                        was_rejected=was_rejected,
                        g_unique_ids=g_unique_ids,
                        probe_unique_ids=probe_unique_ids,
                    )
                )

            # compute uncertainty metrics

            for unc_metric in self.uncertainty_metrics[self.task_type]:
                unc_metrics[gallery_name].update(
                    unc_metric(
                        predicted_id=predicted_id,
                        was_rejected=was_rejected,
                        g_unique_ids=g_unique_ids,
                        probe_unique_ids=probe_unique_ids,
                        predicted_unc=predicted_unc,
                    )
                )

        # aggregate metrics over two galleries
        if len(used_galleries) == 2:
            result_metrics = {}
            result_unc_metrics = {}
            for key in metrics[used_galleries[1]].keys():
                if "osr_metric:" in key:
                    result_metrics[key] = (
                        metrics[used_galleries[0]][key]
                        + metrics[used_galleries[1]][key]
                    ) / 2
                else:
                    result_metrics[key] = metrics[used_galleries[0]][key]
            for key in unc_metrics[used_galleries[1]].keys():
                if "final_auc" in key or "plot_reject" in key:
                    result_unc_metrics[key] = (
                        unc_metrics[used_galleries[0]][key]
                        + unc_metrics[used_galleries[1]][key]
                    ) / 2
                else:
                    result_unc_metrics[key] = unc_metrics[used_galleries[1]][key]
        else:
            result_metrics = metrics[used_galleries[0]]
            result_unc_metrics = unc_metrics[used_galleries[0]]

        return result_metrics, result_unc_metrics

    def run_model_test_verification(
        self,
    ):
        scores = self.distance_function(
            self.template_pooled_emb,
            self.template_pooled_unc,
            self.template_ids,
            self.test_dataset.p1,
            self.test_dataset.p2,
        )

        metrics = {}
        for metric in self.verification_metrics:
            metrics.update(
                metric(
                    scores=scores,
                    labels=self.test_dataset.label,
                )
            )
        return metrics

    def run_model_test_closed_set_identification(self):
        (
            g1_templates_feature,
            g1_template_unc,
            g1_unique_ids,
        ) = self.get_template_subsets(
            self.test_dataset.g1_templates, self.test_dataset.g1_ids
        )
        (
            probe_templates_feature,
            probe_template_unc,
            probe_unique_ids,
        ) = self.get_template_subsets(
            self.test_dataset.probe_templates, self.test_dataset.probe_ids
        )
        is_seen_g1 = np.isin(probe_unique_ids, g1_unique_ids)

        similarity, probe_score = self.distance_function(
            probe_templates_feature[is_seen_g1],
            probe_template_unc[is_seen_g1],
            g1_templates_feature,
            g1_template_unc,
        )

        metrics = {}
        for metric in self.closed_set_identification_metrics:
            metrics.update(
                metric(
                    probe_unique_ids[is_seen_g1],
                    g1_unique_ids,
                    similarity,
                    probe_score,
                )
            )

        if self.use_two_galleries and self.test_dataset.g2_templates.shape != ():
            (
                g2_templates_feature,
                g2_template_unc,
                g2_unique_ids,
            ) = self.get_template_subsets(
                self.test_dataset.g2_templates, self.test_dataset.g2_ids
            )
            print("g2_templates_feature:", g2_templates_feature.shape)  # (1759, 512)
            print(">>>> Gallery 2")
            is_seen_g2 = np.isin(probe_unique_ids, g2_unique_ids)

            similarity, probe_score = self.distance_function(
                probe_templates_feature[is_seen_g2],
                probe_template_unc[is_seen_g2],
                g2_templates_feature,
                g2_template_unc,
            )
            g2_metrics = {}
            for metric in self.closed_set_identification_metrics:
                g2_metrics.update(
                    metric(
                        probe_unique_ids[is_seen_g2],
                        g2_unique_ids,
                        similarity,
                        probe_score,
                    )
                )
            for key in g2_metrics.keys():
                if "cmc" in key:
                    metrics[key] = (metrics[key] + g2_metrics[key]) / 2

        return metrics
