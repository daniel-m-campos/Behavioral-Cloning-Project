from pathlib import Path

from data import get_training_data
from model import train


def main(path: Path):
    model = train(*get_training_data(path, shift=0.1))
    model.save(path / "model.h5")


if __name__ == "__main__":
    main(Path("/home/dcampos/Documents/CarND/data"))
