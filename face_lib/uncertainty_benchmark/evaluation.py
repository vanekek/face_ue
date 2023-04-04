import sys
import hydra
from pathlib import Path

path = str(Path(__file__).parent.parent.parent.absolute())
sys.path.insert(1, path)


@hydra.main(
    config_path=str(Path(".").resolve() / "configs/uncertainty_benchmark"),
    config_name=Path(__file__).stem + "_default",
    version_base="1.2",
)
def run(cfg):
    pass


if __name__ == "__main__":
    run()
