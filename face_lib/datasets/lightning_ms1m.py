import lightning.pytorch as pl
from torch.utils.data import random_split, DataLoader

import sys
sys.path.append('/app')
from face_lib.datasets.ms1m import MXFaceDataset


class MS1M(pl.LightningDataModule):
    def __init__(self, data_ms1m_dir: str, evaluation_configs):
        super().__init__()
        self.data_ms1m_dir = data_ms1m_dir
        self.evaluation_configs = evaluation_configs

    def prepare_data(self):
        pass

    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            self.ms1m_dataset = MXFaceDataset(self.data_ms1m_dir)

        # Assign test dataset for use in dataloader(s)
        # if stage == "test":
        #     self.mnist_test = MNIST(self.data_dir, train=False, transform=self.transform)

        # if stage == "predict":
        #     self.mnist_predict = MNIST(self.data_dir, train=False, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.ms1m_dataset, batch_size=32)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=32)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=32)

    def predict_dataloader(self):
        return DataLoader(self.mnist_predict, batch_size=32)