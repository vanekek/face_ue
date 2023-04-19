from torch.utils.data import DataLoader, Dataset
import cv2

import sys
sys.path.append("/app")

from evaluation.ijb_evals import extract_IJB_data_11, face_align_landmark

class IJB_aligned_images(Dataset):
    def __init__(self, data_ijb_dir: str, subset: str) -> None:
        super().__init__()
        (
            templates,
            medias,
            p1,
            p2,
            label,
            self.img_names,
            self.landmarks,
            face_scores,
        ) = extract_IJB_data_11(data_ijb_dir, subset, force_reload=False)


    def __getitem__(self, index):
        img = self.img_names[index]
        landmark = self.landmarks[index]
        img  = face_align_landmark(cv2.imread(img), landmark)
        img = (img - 127.5) * 0.0078125
        return img.astype("float32")

    def __len__(self):
        return len(self.img_names)