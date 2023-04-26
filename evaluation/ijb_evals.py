#!/usr/bin/env python3
import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from skimage import transform
from sklearn.preprocessing import normalize
from sklearn.metrics import roc_curve, auc
from pathlib import Path
import hydra
import importlib
import sys

path = str(Path(__file__).parent.parent.absolute())
sys.path.insert(1, path)


# https://github.com/deepinsight/insightface/tree/master/recognition/_evaluation_/ijb
class Mxnet_model_interf:
    def __init__(self, model_file, layer="fc1", image_size=(112, 112)):
        import mxnet as mx

        self.mx = mx
        cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
        if len(cvd) > 0 and int(cvd) != -1:
            ctx = [self.mx.gpu(ii) for ii in range(len(cvd.split(",")))]
        else:
            ctx = [self.mx.cpu()]

        prefix, epoch = model_file.split(",")
        print(">>>> loading mxnet model:", prefix, epoch, ctx)
        sym, arg_params, aux_params = self.mx.model.load_checkpoint(prefix, int(epoch))
        all_layers = sym.get_internals()
        sym = all_layers[layer + "_output"]
        model = self.mx.mod.Module(symbol=sym, context=ctx, label_names=None)
        model.bind(data_shapes=[("data", (1, 3, image_size[0], image_size[1]))])
        model.set_params(arg_params, aux_params)
        self.model = model

    def __call__(self, imgs):
        # print(imgs.shape, imgs[0])
        imgs = imgs.transpose(0, 3, 1, 2)
        data = self.mx.nd.array(imgs)
        db = self.mx.io.DataBatch(data=(data,))
        self.model.forward(db, is_train=False)
        emb = self.model.get_outputs()[0].asnumpy()
        return emb


class Torch_model_interf:
    def __init__(self, model_file, image_size=(112, 112)):
        import torch

        self.torch = torch
        cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
        device_name = "cuda:0" if len(cvd) > 0 and int(cvd) != -1 else "cpu"
        self.device = self.torch.device(device_name)
        try:
            self.model = self.torch.jit.load(model_file, map_location=device_name)
        except:
            print(
                "Error: %s is weights only, please load and save the entire model by `torch.jit.save`"
                % model_file
            )
            self.model = None

    def __call__(self, imgs):
        # print(imgs.shape, imgs[0])
        imgs = imgs.transpose(0, 3, 1, 2).copy().astype("float32")
        imgs = (imgs - 127.5) * 0.0078125
        output = self.model(self.torch.from_numpy(imgs).to(self.device).float())
        return output.cpu().detach().numpy()


class ONNX_model_interf:
    def __init__(self, model_file, image_size=(112, 112)):
        import onnxruntime as ort

        ort.set_default_logger_severity(3)
        self.ort_session = ort.InferenceSession(
            model_file, providers=["CUDAExecutionProvider"]
        )
        print(self.ort_session.get_providers())
        exit()
        self.output_names = [self.ort_session.get_outputs()[0].name]
        self.input_name = self.ort_session.get_inputs()[0].name

    def __call__(self, imgs):
        imgs = imgs.transpose(0, 3, 1, 2).astype("float32")
        imgs = (imgs - 127.5) * 0.0078125
        outputs = self.ort_session.run(self.output_names, {self.input_name: imgs})
        return outputs[0]


def keras_model_interf(model_file):
    import tensorflow as tf
    from tensorflow_addons.layers import StochasticDepth

    for gpu in tf.config.experimental.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)

    mm = tf.keras.models.load_model(model_file, compile=False)
    return lambda imgs: mm((tf.cast(imgs, "float32") - 127.5) * 0.0078125).numpy()


def face_align_landmark(img, landmark, image_size=(112, 112), method="similar"):
    tform = (
        transform.AffineTransform()
        if method == "affine"
        else transform.SimilarityTransform()
    )
    src = np.array(
        [
            [38.2946, 51.6963],
            [73.5318, 51.5014],
            [56.0252, 71.7366],
            [41.5493, 92.3655],
            [70.729904, 92.2041],
        ],
        dtype=np.float32,
    )
    tform.estimate(landmark, src)
    # ndimage = transform.warp(img, tform.inverse, output_shape=image_size)
    # ndimage = (ndimage * 255).astype(np.uint8)
    M = tform.params[0:2, :]
    ndimage = cv2.warpAffine(img, M, image_size, borderValue=0.0)
    if len(ndimage.shape) == 2:
        ndimage = np.stack([ndimage, ndimage, ndimage], -1)
    else:
        ndimage = cv2.cvtColor(ndimage, cv2.COLOR_BGR2RGB)
    return ndimage


def read_IJB_meta_columns_to_int(file_path, columns, sep=" ", skiprows=0, header=None):
    # meta = np.loadtxt(file_path, skiprows=skiprows, delimiter=sep)
    meta = pd.read_csv(file_path, sep=sep, skiprows=skiprows, header=header).values
    return (meta[:, ii].astype("int") for ii in columns)


def extract_IJB_data_11(data_path, subset, save_path=None, force_reload=False):
    if save_path == None:
        save_path = os.path.join(data_path, subset + "_backup.npz")
    if not force_reload and os.path.exists(save_path):
        print(">>>> Reload from backup: %s ..." % save_path)
        aa = np.load(save_path)
        return (
            aa["templates"],
            aa["medias"],
            aa["p1"],
            aa["p2"],
            aa["label"],
            aa["img_names"],
            aa["landmarks"],
            aa["face_scores"],
        )

    if subset == "IJBB":
        media_list_path = os.path.join(data_path, "IJBB/meta/ijbb_face_tid_mid.txt")
        pair_list_path = os.path.join(
            data_path, "IJBB/meta/ijbb_template_pair_label.txt"
        )
        img_path = os.path.join(data_path, "IJBB/loose_crop")
        img_list_path = os.path.join(data_path, "IJBB/meta/ijbb_name_5pts_score.txt")
    else:
        media_list_path = os.path.join(data_path, "IJBC/meta/ijbc_face_tid_mid.txt")
        pair_list_path = os.path.join(
            data_path, "IJBC/meta/ijbc_template_pair_label.txt"
        )
        img_path = os.path.join(data_path, "IJBC/loose_crop")
        img_list_path = os.path.join(data_path, "IJBC/meta/ijbc_name_5pts_score.txt")

    print(">>>> Loading templates and medias...")
    templates, medias = read_IJB_meta_columns_to_int(
        media_list_path, columns=[1, 2]
    )  # ['1.jpg', '1', '69544']
    print(
        "templates: %s, medias: %s, unique templates: %s"
        % (templates.shape, medias.shape, np.unique(templates).shape)
    )
    # templates: (227630,), medias: (227630,), unique templates: (12115,)

    print(">>>> Loading pairs...")
    p1, p2, label = read_IJB_meta_columns_to_int(
        pair_list_path, columns=[0, 1, 2]
    )  # ['1', '11065', '1']
    print("p1: %s, unique p1: %s" % (p1.shape, np.unique(p1).shape))
    print("p2: %s, unique p2: %s" % (p2.shape, np.unique(p2).shape))
    print(
        "label: %s, label value counts: %s"
        % (label.shape, dict(zip(*np.unique(label, return_counts=True))))
    )
    # p1: (8010270,), unique p1: (1845,)
    # p2: (8010270,), unique p2: (10270,) # 10270 + 1845 = 12115 --> np.unique(templates).shape
    # label: (8010270,), label value counts: {0: 8000000, 1: 10270}

    print(">>>> Loading images...")
    with open(img_list_path, "r") as ff:
        # 1.jpg 46.060 62.026 87.785 60.323 68.851 77.656 52.162 99.875 86.450 98.648 0.999
        img_records = np.array([ii.strip().split(" ") for ii in ff.readlines()])

    img_names = np.array([os.path.join(img_path, ii) for ii in img_records[:, 0]])
    landmarks = img_records[:, 1:-1].astype("float32").reshape(-1, 5, 2)
    face_scores = img_records[:, -1].astype("float32")
    print(
        "img_names: %s, landmarks: %s, face_scores: %s"
        % (img_names.shape, landmarks.shape, face_scores.shape)
    )
    # img_names: (227630,), landmarks: (227630, 5, 2), face_scores: (227630,)
    print(
        "face_scores value counts:", dict(zip(*np.histogram(face_scores, bins=9)[::-1]))
    )
    # {0.1: 2515, 0.2: 0, 0.3: 62, 0.4: 94, 0.5: 136, 0.6: 197, 0.7: 291, 0.8: 538, 0.9: 223797}

    print(">>>> Saving backup to: %s ..." % save_path)
    np.savez(
        save_path,
        templates=templates,
        medias=medias,
        p1=p1,
        p2=p2,
        label=label,
        img_names=img_names,
        landmarks=landmarks,
        face_scores=face_scores,
    )
    print()
    return templates, medias, p1, p2, label, img_names, landmarks, face_scores


def extract_gallery_prob_data(data_path, subset, save_path=None, force_reload=False):
    if save_path == None:
        save_path = os.path.join(data_path, subset + "_gallery_prob_backup.npz")
    if not force_reload and os.path.exists(save_path):
        print(">>>> Reload from backup: %s ..." % save_path)
        aa = np.load(save_path)
        return (
            aa["s1_templates"],
            aa["s1_subject_ids"],
            aa["s2_templates"],
            aa["s2_subject_ids"],
            aa["probe_mixed_templates"],
            aa["probe_mixed_subject_ids"],
        )

    if subset == "IJBC":
        meta_dir = os.path.join(data_path, "IJBC/meta")
        gallery_s1_record = os.path.join(meta_dir, "ijbc_1N_gallery_G1.csv")
        gallery_s2_record = os.path.join(meta_dir, "ijbc_1N_gallery_G2.csv")
        probe_mixed_record = os.path.join(meta_dir, "ijbc_1N_probe_mixed.csv")
    else:
        meta_dir = os.path.join(data_path, "IJBB/meta")
        gallery_s1_record = os.path.join(meta_dir, "ijbb_1N_gallery_S1.csv")
        gallery_s2_record = os.path.join(meta_dir, "ijbb_1N_gallery_S2.csv")
        probe_mixed_record = os.path.join(meta_dir, "ijbb_1N_probe_mixed.csv")

    print(">>>> Loading gallery feature...")
    s1_templates, s1_subject_ids = read_IJB_meta_columns_to_int(
        gallery_s1_record, columns=[0, 1], skiprows=1, sep=","
    )
    s2_templates, s2_subject_ids = read_IJB_meta_columns_to_int(
        gallery_s2_record, columns=[0, 1], skiprows=1, sep=","
    )
    print(
        "s1 gallery: %s, ids: %s, unique: %s"
        % (s1_templates.shape, s1_subject_ids.shape, np.unique(s1_templates).shape)
    )
    print(
        "s2 gallery: %s, ids: %s, unique: %s"
        % (s2_templates.shape, s2_subject_ids.shape, np.unique(s2_templates).shape)
    )

    print(">>>> Loading prope feature...")
    probe_mixed_templates, probe_mixed_subject_ids = read_IJB_meta_columns_to_int(
        probe_mixed_record, columns=[0, 1], skiprows=1, sep=","
    )
    print(
        "probe_mixed_templates: %s, unique: %s"
        % (probe_mixed_templates.shape, np.unique(probe_mixed_templates).shape)
    )
    print(
        "probe_mixed_subject_ids: %s, unique: %s"
        % (probe_mixed_subject_ids.shape, np.unique(probe_mixed_subject_ids).shape)
    )

    print(">>>> Saving backup to: %s ..." % save_path)
    np.savez(
        save_path,
        s1_templates=s1_templates,
        s1_subject_ids=s1_subject_ids,
        s2_templates=s2_templates,
        s2_subject_ids=s2_subject_ids,
        probe_mixed_templates=probe_mixed_templates,
        probe_mixed_subject_ids=probe_mixed_subject_ids,
    )
    print()
    return (
        s1_templates,
        s1_subject_ids,
        s2_templates,
        s2_subject_ids,
        probe_mixed_templates,
        probe_mixed_subject_ids,
    )


def get_embeddings(model_interf, img_names, landmarks, batch_size=64, flip=True):
    steps = int(np.ceil(len(img_names) / batch_size))
    embs, embs_f = [], []
    for batch_id in tqdm(
        range(0, len(img_names), batch_size), "Embedding", total=steps
    ):
        batch_imgs, batch_landmarks = (
            img_names[batch_id : batch_id + batch_size],
            landmarks[batch_id : batch_id + batch_size],
        )
        ndimages = [
            face_align_landmark(cv2.imread(img), landmark)
            for img, landmark in zip(batch_imgs, batch_landmarks)
        ]
        ndimages = np.stack(ndimages)
        embs.extend(model_interf(ndimages))
        if flip:
            embs_f.extend(model_interf(ndimages[:, :, ::-1, :]))
    return np.array(embs), np.array(embs_f)


def process_embeddings(
    embs,
    embs_f=[],
    use_flip_test=True,
    use_norm_score=False,
    use_detector_score=True,
    face_scores=None,
):
    print(
        ">>>> process_embeddings: Norm {}, Detect_score {}, Flip {}".format(
            use_norm_score, use_detector_score, use_flip_test
        )
    )
    if use_flip_test and len(embs_f) != 0:
        embs = embs + embs_f
    if use_norm_score:
        embs = normalize(embs)
    if use_detector_score and face_scores is not None:
        embs = embs * np.expand_dims(face_scores, -1)
    return embs


def image2template_feature(
    img_feats=None, templates=None, medias=None, choose_templates=None, choose_ids=None
):
    if choose_templates is not None:  # 1:N
        unique_templates, indices = np.unique(choose_templates, return_index=True)
        unique_subjectids = choose_ids[indices]
    else:  # 1:1
        unique_templates = np.unique(templates)
        unique_subjectids = None

    # template_feats = np.zeros((len(unique_templates), img_feats.shape[1]), dtype=img_feats.dtype)
    template_feats = np.zeros((len(unique_templates), img_feats.shape[1]))
    for count_template, uqt in tqdm(
        enumerate(unique_templates),
        "Extract template feature",
        total=len(unique_templates),
    ):
        (ind_t,) = np.where(templates == uqt)
        face_norm_feats = img_feats[ind_t]
        face_medias = medias[ind_t]
        unique_medias, unique_media_counts = np.unique(face_medias, return_counts=True)
        media_norm_feats = []
        for u, ct in zip(unique_medias, unique_media_counts):
            (ind_m,) = np.where(face_medias == u)
            if ct == 1:
                media_norm_feats += [face_norm_feats[ind_m]]
            else:  # image features from the same video will be aggregated into one feature
                media_norm_feats += [np.mean(face_norm_feats[ind_m], 0, keepdims=True)]
        media_norm_feats = np.array(media_norm_feats)
        # media_norm_feats = media_norm_feats / np.sqrt(np.sum(media_norm_feats ** 2, -1, keepdims=True))
        template_feats[count_template] = np.sum(media_norm_feats, 0)
    template_norm_feats = normalize(template_feats)
    return template_norm_feats, unique_templates, unique_subjectids


def verification_11(
    template_norm_feats=None, unique_templates=None, p1=None, p2=None, batch_size=10000
):
    try:
        print(">>>> Trying cupy.")
        import cupy as cp

        template_norm_feats = cp.array(template_norm_feats)
        score_func = lambda feat1, feat2: cp.sum(feat1 * feat2, axis=-1).get()
        test = score_func(
            template_norm_feats[:batch_size], template_norm_feats[:batch_size]
        )
    except:
        score_func = lambda feat1, feat2: np.sum(feat1 * feat2, -1)

    template2id = np.zeros(max(unique_templates) + 1, dtype=int)
    template2id[unique_templates] = np.arange(len(unique_templates))

    steps = int(np.ceil(len(p1) / batch_size))
    score = []
    for id in tqdm(range(steps), "Verification"):
        feat1 = template_norm_feats[
            template2id[p1[id * batch_size : (id + 1) * batch_size]]
        ]
        feat2 = template_norm_feats[
            template2id[p2[id * batch_size : (id + 1) * batch_size]]
        ]
        score.extend(score_func(feat1, feat2))
    return np.array(score)


class IJB_test:
    def __init__(
        self,
        model_file,
        data_path,
        subset,
        evaluation_1N_function,
        batch_size=64,
        force_reload=False,
        restore_embs=None,
    ):
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
        template_norm_feats, unique_templates, _ = image2template_feature(
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
        two_galleries = False

        fars_cal = [10**ii for ii in np.arange(-4, 0, 4 / npoints)] + [
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
            use_flip_test=True,
            use_norm_score=False,
            use_detector_score=True,
            face_scores=self.face_scores,
        )
        (
            g1_templates_feature,
            g1_unique_templates,
            g1_unique_ids,
        ) = image2template_feature(
            img_input_feats, self.templates, self.medias, g1_templates, g1_ids
        )
        if two_galleries:
            (
                g2_templates_feature,
                g2_unique_templates,
                g2_unique_ids,
            ) = image2template_feature(
                img_input_feats, self.templates, self.medias, g2_templates, g2_ids
            )
        probe_mixed_templates_feature_path = (
            f"/app/cache/template_cache/probe_aggr_{self.subset}"
        )
        if Path(probe_mixed_templates_feature_path + "_feature.npy").is_file():
            probe_mixed_templates_feature = np.load(
                probe_mixed_templates_feature_path + "_feature.npy"
            )
            probe_mixed_unique_subject_ids = np.load(
                probe_mixed_templates_feature_path + "_subject_ids.npy"
            )
        else:
            (
                probe_mixed_templates_feature,
                probe_mixed_unique_templates,
                probe_mixed_unique_subject_ids,
            ) = image2template_feature(
                img_input_feats,
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
                probe_mixed_templates_feature_path + "_subject_ids.npy",
                probe_mixed_unique_subject_ids,
            )
        print("g1_templates_feature:", g1_templates_feature.shape)  # (1772, 512)

        if two_galleries:
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
            g1_templates_feature,
            probe_mixed_unique_subject_ids,
            g1_unique_ids,
            fars_cal,
        )

        if two_galleries:
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
                g2_templates_feature,
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
            show_result = {}
            for id, far in enumerate(fars_cal):
                if id in fars_show_idx:
                    show_result.setdefault("far", []).append(far)
                    show_result.setdefault("g1_tpir", []).append(g1_recalls[id])
                    show_result.setdefault("g1_thresh", []).append(g1_threshes[id])
                    show_result.setdefault("g2_tpir", []).append(g2_recalls[id])
                    show_result.setdefault("g2_thresh", []).append(g2_threshes[id])
                    show_result.setdefault("mean_tpir", []).append(mean_tpirs[id])
            print(pd.DataFrame(show_result).set_index("far").to_markdown())
        else:
            mean_tpirs = np.array(g1_recalls)
        return fars_cal, mean_tpirs, None, None  # g1_cmc_scores, g2_cmc_scores


def plot_roc_and_calculate_tpr(scores, names=None, label=None):
    print(">>>> plot roc and calculate tpr...")
    score_dict = {}
    for id, score in enumerate(scores):
        name = None if names is None else names[id]
        if isinstance(score, str) and score.endswith(".npz"):
            aa = np.load(score)
            score = aa.get("scores", [])
            label = aa["label"] if label is None and "label" in aa else label
            score_name = aa.get("names", [])
            for ss, nn in zip(score, score_name):
                score_dict[nn] = ss
        elif isinstance(score, str) and score.endswith(".npy"):
            name = (
                name
                if name is not None
                else os.path.splitext(os.path.basename(score))[0]
            )
            score_dict[name] = np.load(score)
        elif isinstance(score, str) and score.endswith(".txt"):
            # IJB meta data like ijbb_template_pair_label.txt
            label = pd.read_csv(score, sep=" ", header=None).values[:, 2]
        else:
            name = name if name is not None else str(id)
            score_dict[name] = score
    if label is None:
        print("Error: Label data is not provided")
        return None, None

    x_labels = [10 ** (-ii) for ii in range(1, 7)[::-1]]
    fpr_dict, tpr_dict, roc_auc_dict, tpr_result = {}, {}, {}, {}
    for name, score in score_dict.items():
        fpr, tpr, _ = roc_curve(label, score)
        roc_auc = auc(fpr, tpr)
        fpr, tpr = np.flipud(fpr), np.flipud(tpr)  # select largest tpr at same fpr
        tpr_result[name] = [tpr[np.argmin(abs(fpr - ii))] for ii in x_labels]
        fpr_dict[name], tpr_dict[name], roc_auc_dict[name] = fpr, tpr, roc_auc
    tpr_result_df = pd.DataFrame(tpr_result, index=x_labels).T
    tpr_result_df["AUC"] = pd.Series(roc_auc_dict)
    tpr_result_df.columns.name = "Methods"
    print(tpr_result_df.to_markdown())
    # print(tpr_result_df)

    try:
        import matplotlib.pyplot as plt

        fig = plt.figure()
        for name in score_dict:
            plt.plot(
                fpr_dict[name],
                tpr_dict[name],
                lw=1,
                label="[%s (AUC = %0.4f%%)]" % (name, roc_auc_dict[name] * 100),
            )
        title = (
            "ROC on IJB" + name.split("IJB")[-1][0] if "IJB" in name else "ROC on IJB"
        )

        plt.xlim([10**-6, 0.1])
        plt.xscale("log")
        plt.xticks(x_labels)
        plt.xlabel("False Positive Rate")
        plt.ylim([0.3, 1.0])
        plt.yticks(np.linspace(0.3, 1.0, 8, endpoint=True))
        plt.ylabel("True Positive Rate")

        plt.grid(linestyle="--", linewidth=1)
        plt.title(title)
        plt.legend(loc="lower right", fontsize="x-small")
        plt.tight_layout()
        plt.show()
    except:
        print("matplotlib plot failed")
        fig = None

    return tpr_result_df, fig


def plot_dir_far_cmc_scores(scores, names=None):
    try:
        import matplotlib.pyplot as plt

        fig = plt.figure()
        for id, score in enumerate(scores):
            name = None if names is None else names[id]
            if isinstance(score, str) and score.endswith(".npz"):
                aa = np.load(score)
                score, name = aa.get("scores")[0], aa.get("names")[0]
            fars, tpirs = score[0], score[1]
            name = name if name is not None else str(id)

            auc_value = auc(fars, tpirs)
            label = "[%s (AUC = %0.4f%%)]" % (name, auc_value * 100)
            plt.plot(fars, tpirs, lw=1, label=label)

        plt.xlabel("False Alarm Rate")
        plt.xlim([0.0001, 1])
        plt.xscale("log")
        plt.ylabel("Detection & Identification Rate (%)")
        plt.ylim([0, 1])

        plt.grid(linestyle="--", linewidth=1)
        plt.legend(fontsize="x-small")
        plt.tight_layout()
    except:
        print("matplotlib plot failed")
        fig = None

    return fig


@hydra.main(
    config_path=str(Path(".").resolve() / "configs/uncertainty_benchmark"),
    config_name=Path(__file__).stem,
    version_base="1.2",
)
def main(cfg):
    save_name = os.path.splitext(os.path.basename(cfg.save_result))[0]
    save_items = {}
    save_path = os.path.dirname(cfg.save_result)
    if len(save_path) != 0 and not os.path.exists(save_path):
        os.makedirs(save_path)
    module_name_parts = cfg.evaluation_1N_function.class_path.split(".")
    module_path = ".".join(module_name_parts[:-1])
    class_name = module_name_parts[-1]
    one_to_N_eval_function = getattr(importlib.import_module(module_path), class_name)(
        **cfg.evaluation_1N_function.init_args
    )
    tt = IJB_test(
        model_file=None,
        data_path=cfg.data_path,
        subset=cfg.subset,
        evaluation_1N_function=one_to_N_eval_function,
        batch_size=cfg.batch_size,
        force_reload=False,
        restore_embs=cfg.restore_embs,
    )

    if cfg.is_one_2_N:  # 1:N test
        fars, tpirs, _, _ = tt.run_model_test_1N()
        scores = [(fars, tpirs)]
        names = [save_name]
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

    np.savez(cfg.save_result, **save_items)

    if cfg.is_one_2_N:
        pass
        fig = plot_dir_far_cmc_scores(scores=scores, names=names)
        fig.savefig(Path(cfg.exp_dir) / "di_far_plot.png")
    else:
        plot_roc_and_calculate_tpr(scores, names=names, label=label)


if __name__ == "__main__":
    main()
