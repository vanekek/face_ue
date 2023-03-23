import torch

import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import mxnet as mx
import numbers
import numpy as np
import os
from pathlib import Path

import sys

sys.path.append("/app")
from face_lib.utils.imageprocessing import preprocess
from face_lib.datasets import IJBDataset


class MXFaceDataset(Dataset):
    def __init__(self, root_dir):
        super(MXFaceDataset, self).__init__()
        self.transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )
        self.root_dir = root_dir
        path_imgrec = os.path.join(root_dir, "train.rec")
        path_imgidx = os.path.join(root_dir, "train.idx")
        self.imgrec = mx.recordio.MXIndexedRecordIO(path_imgidx, path_imgrec, "r")
        s = self.imgrec.read_idx(0)
        header, _ = mx.recordio.unpack(s)
        if header.flag > 0:
            self.header0 = (int(header.label[0]), int(header.label[1]))
            self.imgidx = np.array(range(1, int(header.label[0])))
        else:
            self.imgidx = np.array(list(self.imgrec.keys))

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
        return sample, label

    def __len__(self):
        return len(self.imgidx)


class IJBC_images(Dataset):
    def __init__(self, data_ijbc_dir) -> None:
        super().__init__()
        testset = IJBDataset(data_ijbc_dir)
        self.image_paths = testset["abspath"].values
        self.short_paths = ["/".join(Path(p).parts[-2:]) for p in self.image_paths]

        self.proc_func = lambda images: preprocess(
            images, [112, 112], is_training=False
        )

    def __getitem__(self, index):
        return self.proc_func([self.image_paths[index]])[0], self.short_paths[index]

    def __len__(self):
        return len(self.image_paths)


class SCF_DataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_train_dir: str,
        data_predict_dir: str,
        batch_size: int,
        num_workers: int,
    ):
        super().__init__()
        self.data_train_dir = data_train_dir
        self.data_predict_dir = data_predict_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare_data(self):
        pass

    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            self.ms1m_dataset = MXFaceDataset(self.data_train_dir)
            # self.ms1m_dataset = torch.utils.data.Subset(self.ms1m_dataset, np.random.choice(len(self.ms1m_dataset), 5000, replace=False))
        # Assign test dataset for use in dataloader(s)
        # if stage == "test":
        #     self.mnist_test = MNIST(self.data_dir, train=False, transform=self.transform)

        if stage == "predict":
            self.ijbc_dataset = IJBC_images(self.data_predict_dir)
            # self.ijbc_dataset = torch.utils.data.Subset(self.ijbc_dataset, np.random.choice(len(self.ijbc_dataset), 500, replace=False))

    def train_dataloader(self):
        return DataLoader(
            self.ms1m_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=self.num_workers,
        )

    # def val_dataloader(self):
    #     return DataLoader(self.mnist_val, batch_size=32)

    # def test_dataloader(self):
    #     return DataLoader(self.mnist_test, batch_size=32)

    def predict_dataloader(self):
        return DataLoader(
            self.ijbc_dataset,
            batch_size=self.batch_size,
            drop_last=False,
            shuffle=False,
            num_workers=self.num_workers,
        )
