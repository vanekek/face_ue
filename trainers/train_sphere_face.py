
import sys
sys.path.append("/app")


from lightning.pytorch.cli import LightningCLI

# simple demo classes for your convenience
from lightning.pytorch.demos.boring_classes import DemoModel


def cli_main():
    cli = LightningCLI(DemoModel)


if __name__ == "__main__":
    cli_main()