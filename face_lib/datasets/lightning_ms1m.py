import torch

import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import mxnet as mx
import numbers
import numpy as np
import os


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


class MS1M(pl.LightningDataModule):
    def __init__(self, data_ms1m_dir: str, batch_size: int, num_workers: int):
        super().__init__()
        self.data_ms1m_dir = data_ms1m_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare_data(self):
        pass

    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            self.ms1m_dataset = MXFaceDataset(self.data_ms1m_dir)
            # self.ms1m_dataset = torch.utils.data.Subset(self.ms1m_dataset, np.random.choice(len(self.ms1m_dataset), 5000, replace=False))
        # Assign test dataset for use in dataloader(s)
        # if stage == "test":
        #     self.mnist_test = MNIST(self.data_dir, train=False, transform=self.transform)

        # if stage == "predict":
        #     self.mnist_predict = MNIST(self.data_dir, train=False, transform=self.transform)

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

    # def predict_dataloader(self):
    #     return DataLoader(self.mnist_predict, batch_size=32)
