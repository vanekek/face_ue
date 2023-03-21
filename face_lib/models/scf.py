import torch
from lightning import LightningModule

class Encoder(torch.nn.Module):
    ...


class Decoder(torch.nn.Module):
    ...


class AutoEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        return self.decoder(self.encoder(x))


class SphereConfidenceFace(LightningModule):
    def __init__(self, backbone, head):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x):
        backbone_outputs = self.backbone(x)
        log_kappa = self.head(backbone_outputs['bottleneck_feature'])
        return log_kappa

    def training_step(self, batch, batch_idx):
        x, y = batch
        log_kappa = self(x)
        loss = 1
        return loss

    def validation_step(self, batch, batch_idx):
        self._shared_eval(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        self._shared_eval(batch, batch_idx, "test")

    def _shared_eval(self, batch, batch_idx, prefix):
        x, _ = batch
        x_hat = self.auto_encoder(x)
        loss = self.metric(x, x_hat)
        self.log(f"{prefix}_loss", loss)