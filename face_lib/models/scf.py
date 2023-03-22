import torch
from lightning import LightningModule
import importlib


class SphereConfidenceFace(LightningModule):
    def __init__(
        self,
        backbone: torch.nn.Module,
        head: torch.nn.Module,
        scf_loss: torch.nn.Module,
        optimizer,
        scheduler_params,
        softmax_weights_path: str,
        radius: int,
    ):
        super().__init__()
        self.backbone = backbone
        self.head = head
        self.scf_loss = scf_loss
        self.optimizer = optimizer
        self.scheduler_params = scheduler_params
        self.softmax_weights = torch.load(softmax_weights_path)
        softmax_weights_norm = torch.norm(
            self.softmax_weights, dim=1, keepdim=True
        )  # [N, 512]
        self.softmax_weights = (
            self.softmax_weights / softmax_weights_norm * radius
        )  # $ w_c \in rS^{d-1} $

    def forward(self, x):
        backbone_outputs = self.backbone(x)
        log_kappa = self.head(backbone_outputs["bottleneck_feature"])
        return backbone_outputs["feature"], log_kappa

    def training_step(self, batch, batch_idx):
        images, labels = batch
        feature, log_kappa = self(images)
        kappa = torch.exp(log_kappa)
        wc = self.softmax_weights[labels, :]
        losses, l1, l2, l3 = self.scf_loss(feature, kappa, wc)

        kappa_mean = kappa.mean()
        total_loss = losses.mean()
        neg_kappa_times_cos_theta = l1.mean()
        neg_dim_scalar_times_log_kappa = l2.mean()
        log_iv_kappa = l3.mean()

        self.log("train_loss", total_loss.item())
        self.log("kappa", kappa_mean.item())
        self.log("total_loss", total_loss.item())
        self.log("neg_kappa_times_cos_theta", neg_kappa_times_cos_theta.item())
        self.log(
            "neg_dim_scalar_times_log_kappa", neg_dim_scalar_times_log_kappa.item()
        )
        self.log("log_iv_kappa", log_iv_kappa.item())

        return total_loss

    def configure_optimizers(self):
        optimizer = self.optimizer(self.head.parameters())
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": getattr(
                    importlib.import_module("torch.optim.lr_scheduler"),
                    self.scheduler_params.scheduler,
                )(optimizer, **self.scheduler_params.params),
                "monitor": "train_loss",
            },
        }

    # def validation_step(self, batch, batch_idx):
    #     self._shared_eval(batch, batch_idx, "val")

    # def test_step(self, batch, batch_idx):
    #     self._shared_eval(batch, batch_idx, "test")

    # def _shared_eval(self, batch, batch_idx, prefix):
    #     x, _ = batch
    #     x_hat = self.auto_encoder(x)
    #     loss = self.metric(x, x_hat)
    #     self.log(f"{prefix}_loss", loss)
