import sys

sys.path.append("/app")

# code based on original repo: https://github.com/MathsShen/SCF
# main config for training: configs/hydra/train_sphere_face.yaml
from lightning.pytorch.cli import LightningCLI

# simple demo classes for your convenience
from face_lib.models.scf import SphereConfidenceFace


def cli_main():
    cli = LightningCLI(SphereConfidenceFace)


if __name__ == "__main__":
    cli_main()
