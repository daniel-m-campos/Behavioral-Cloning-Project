from pathlib import Path

from tensorflow import keras

import data
from model import train


def main(path: Path):
    trainer, validator = data.create_validation_generators(
        path, validation_split=0.2, shift=0.125, batch_size=100
    )
    model = train(trainer, validator, epochs=2)
    model.save(path / "model.h5")


def plot_model(path: Path):
    model = keras.models.load_model(path / "model.h5")
    keras.utils.plot_model(model, to_file="../../img/model.png", show_shapes=True)


if __name__ == "__main__":
    main(Path("/home/dcampos/Documents/CarND/data"))
