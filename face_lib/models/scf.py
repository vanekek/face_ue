import torch
from lightning import LightningModule

class SphereConfidenceFace(LightningModule):
    def __init__(self, backbone: torch.nn.Module, head: torch.nn.Module, softmax_weights_path: str, radius: int):
        super().__init__()
        self.backbone = backbone
        self.head = head
        self.softmax_weights = torch.load(softmax_weights_path)
        softmax_weights_norm = torch.norm(self.softmax_weights, dim=1, keepdim=True) #[N, 512]
        self.softmax_weights = self.softmax_weights / softmax_weights_norm * radius # $ w_c \in rS^{d-1} $


    def forward(self, x):
        backbone_outputs = self.backbone(x)
        log_kappa = self.head(backbone_outputs['bottleneck_feature'])
        return backbone_outputs['feature'], log_kappa

    def training_step(self, batch, batch_idx):
        images, labels = batch
        feature, log_kappa = self(images)
        kappa = torch.exp(log_kappa)
        wc = self.softmax_weights[labels, :]
        losses, l1, l2, l3 = kl(mu, kappa, wc)
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