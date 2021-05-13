from pathlib import Path

from tensorflow import keras

from behavior_cloning import data
from behavior_cloning.model import train


def plot_model(path: Path):
    model = keras.models.load_model(path / "model.h5")
    keras.utils.plot_model(model, to_file="../../img/model2.png", show_shapes=True)


def main(path: Path):
    trainer, validator = data.create_validation_generators(
        path, validation_split=0.2, shift=0.125, batch_size=100
    )
    model = train(trainer, validator, epochs=10)
    model.save(path / "model.h5")
    plot_model(path / "model.png")


if __name__ == "__main__":
    main(Path("../../data/track2"))
