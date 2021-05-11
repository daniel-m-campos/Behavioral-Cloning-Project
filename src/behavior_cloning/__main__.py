from pathlib import Path

from data import read, convert
from model import train


def main(path: Path):
    model = train(*convert(read(path)))
    model.save(path / "model.h5")


if __name__ == "__main__":
    main(Path("/home/dcampos/Documents/CarND/data"))
