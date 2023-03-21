
import sys
sys.path.append("/app")


from lightning.pytorch.cli import LightningCLI

# simple demo classes for your convenience
from face_lib.models.scf import SphereConfidenceFace


def cli_main():
    cli = LightningCLI(SphereConfidenceFace)


if __name__ == "__main__":
    cli_main()