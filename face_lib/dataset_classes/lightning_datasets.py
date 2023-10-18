import torch

import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import mxnet as mx
import numbers
from pathlib import Path
from tqdm import tqdm
import numpy as np
import os

class MXFaceDataset(Dataset):
    def __init__(self, root_dir, test=False, num_classes=0):
        """
        ArcFace loader
        https://github.com/deepinsight/insightface/blob/master/recognition/arcface_torch/dataset.py
        """
        super(MXFaceDataset, self).__init__()
        self.num_classes = num_classes
        self.test = test
        if self.test:
            self.transform = transforms.Compose(
                [
                    transforms.ToPILImage(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                ]
            )
        else:
            self.transform = transforms.Compose(
                [
                    transforms.ToPILImage(),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                ]
            )

                # load pictures
        self.root_dir = root_dir
        path_imgrec = os.path.join(root_dir, "train.rec")
        path_imgidx = os.path.join(root_dir, "train.idx")
        self.imgrec = mx.recordio.MXIndexedRecordIO(path_imgidx, path_imgrec, "r")
        s = self.imgrec.read_idx(0)
        header, _ = mx.recordio.unpack(s)

        self.imgrec = mx.recordio.MXIndexedRecordIO(path_imgidx, path_imgrec, "r")
        s = self.imgrec.read_idx(0)
        header, _ = mx.recordio.unpack(s)

        self.imgidx = np.array(range(1, int(header.label[0])))

        # load or create labels
        labels_path = Path(root_dir) / "labels.npy"
        if labels_path.is_file():
            self.labels = np.load(labels_path)
        else:
            print('Listing labels...')
            labels = []
            for i in range(len(self.imgidx)):
                idx = self.imgidx[i]
                s = self.imgrec.read_idx(idx)
                header, img = mx.recordio.unpack(s)
                label = header.label
                labels.append(int(label))
            self.labels = np.array(labels)
            # save labels
            np.save(labels_path, self.labels)

        if num_classes > 0:
            seed = 0
            min_size = 30
            image_idx_path = Path(root_dir) / f"image_idx_{num_classes}-classes_{seed}-seed_{min_size}-min-class-size.npy"
            self.image_label_path = Path(root_dir) / f"image_label_{num_classes}-classes_{seed}-seed_{min_size}-min-class-size.npy"
            if image_idx_path.is_file():
                self.imgidx = np.load(image_idx_path)
                self.labels = np.load(self.image_label_path)
            else:
                print(f'Listing images of {num_classes} random classes...')
                rng = np.random.default_rng(seed)
                unique_labels, unique_counts = np.unique(self.labels, return_counts=True)
                unique_labels_thresh = unique_labels[unique_counts > min_size]
                selected_classes = rng.choice(unique_labels_thresh, num_classes, replace=False)
                imgidx_short = []
                labels_short = []
                for selected_class in tqdm(selected_classes):
                    index = (self.labels == selected_class)
                    imgidx_short.extend(list(self.imgidx[index]))
                    labels_short.extend(list(self.labels[index]))
                self.imgidx = np.array(imgidx_short)
                self.labels = np.array(labels_short)
                np.save(image_idx_path, self.imgidx)
                np.save(self.image_label_path, self.labels)
    def create_identification_meta(self, identification_ds_path: Path):

        pass

    def __getitem__(self, index):
        idx = self.imgidx[index]
        s = self.imgrec.read_idx(idx)
        header, img = mx.recordio.unpack(s)
        label = header.label
        if not isinstance(label, numbers.Number):
            label = label[0]
        label = torch.tensor(label, dtype=torch.long)
        sample = mx.image.imdecode(img).asnumpy()
        if self.transform is not None:
            sample = self.transform(sample)
        if self.test:
            return sample
        else:
            return sample, label

    def __len__(self):
        return len(self.imgidx)


class UncertaintyDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_train_dir: str,
        predict_dataset: Dataset,
        batch_size: int,
        num_workers: int,
    ):
        super().__init__()
        self.data_train_dir = data_train_dir
        self.predict_dataset = predict_dataset
        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare_data(self):
        pass

    def setup(self, stage: str):
        # Assign train/predict datasets for use in dataloaders
        if stage == "fit":
            self.ms1m_dataset = MXFaceDataset(self.data_train_dir)
            # self.ms1m_dataset = torch.utils.data.Subset(self.ms1m_dataset, np.random.choice(len(self.ms1m_dataset), 5000, replace=False))

        if stage == "predict":
            pass
            # self.ijb_dataset = IJB_aligned_images(self.data_predict_dir, self.data_predict_subset)
            # self.predict_dataset = torch.utils.data.Subset(self.predict_dataset, np.random.choice(len(self.predict_dataset), 5000, replace=False))

    def train_dataloader(self):
        return DataLoader(
            self.ms1m_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=self.num_workers,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.predict_dataset,
            batch_size=self.batch_size,
            drop_last=False,
            shuffle=False,
            num_workers=self.num_workers,
        )
