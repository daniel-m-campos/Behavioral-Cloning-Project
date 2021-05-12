from pathlib import Path

import data
from model import train


def main(path: Path):
    trainer, validator = data.create_validation_generators(
        path, validation_split=0.2, shift=0.125, batch_size=100
    )
    model = train(trainer, validator, epochs=2)
    model.save(path / "model.h5")


if __name__ == "__main__":
    main(Path("/home/dcampos/Documents/CarND/data"))
