from torch.utils.data import DataLoader, Dataset
import cv2
import numpy as np
import sys

sys.path.append("/app")

from evaluation.embeddings import face_align_landmark
from evaluation.data_tools import extract_meta_data


class IJB_aligned_images(Dataset):
    def __init__(
        self, dataset_path: str, dataset_name: str, norm_image: bool = False
    ) -> None:
        super().__init__()
        self.norm_image = norm_image
        (
            templates,
            medias,
            p1,
            p2,
            label,
            self.img_names,
            self.landmarks,
            face_scores,
        ) = extract_meta_data(dataset_path, dataset_name)

    def __getitem__(self, index):
        img = self.img_names[index]
        landmark = self.landmarks[index]
        img = face_align_landmark(cv2.imread(str(img)), landmark)
        img = (img - 127.5) * 0.0078125
        if self.norm_image:
            img = (
                img - np.array([0.5, 0.5, 0.5])[np.newaxis, np.newaxis, :]
            ) / np.array([0.5, 0.5, 0.5])[np.newaxis, np.newaxis, :]

        return img.astype("float32")

    def __len__(self):
        return len(self.img_names)
