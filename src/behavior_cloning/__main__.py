import itertools
import math
import os
from pathlib import Path

from tensorflow import keras

from behavior_cloning import data
from behavior_cloning.model import train


def plot_model(path: Path):
    model = keras.models.load_model(path / "model.h5")
    to_file = (path / "model.png").absolute()
    keras.utils.plot_model(model, to_file=to_file, show_shapes=True)


def get_num_files(path: Path):
    return len(
        [name for name in os.listdir(path) if os.path.isfile(os.path.join(path, name))]
    )


def main(
    path: Path,
    epochs: int,
    batch_size: int,
    validation_split: float = 0.2,
    shift: float = 0.125,
):
    trainer, validator = data.create_validation_generators(
        path, validation_split=validation_split, shift=shift, batch_size=batch_size
    )
    trainer = itertools.cycle(trainer)
    validator = itertools.cycle(validator)
    steps_per_epoch = get_num_files(path / "IMG") / batch_size
    model = train(
        trainer,
        validator,
        epochs=epochs,
        steps_per_epoch=math.ceil(steps_per_epoch * (1 - validation_split)),
        validation_steps=math.ceil(steps_per_epoch * validation_split),
    )
    model.save(path / "model.h5")
    plot_model(path)


if __name__ == "__main__":
    main(Path("../../data/track2"), epochs=2, batch_size=100)
