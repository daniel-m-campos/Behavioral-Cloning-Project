import itertools
import math
import os
import pickle
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
    batch_size: int = None,
    validation_split: float = 0.2,
    shift: float = 0.125,
):
    steps_per_epoch, validation_steps = None, None
    if batch_size is not None:
        trainer, validator = data.create_validation_generators(
            path, validation_split=validation_split, shift=shift, batch_size=batch_size
        )
        trainer = itertools.cycle(trainer)
        validator = itertools.cycle(validator)
        steps = get_num_files(path / "IMG") / batch_size
        steps_per_epoch = math.ceil(steps * (1 - validation_split))
        validation_steps = math.ceil(steps * validation_split)
        validation_split = None
    else:
        trainer = data.create_data_generator(path, shift=shift)
        validator = None
    model, history = train(
        trainer,
        validator,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        validation_split=validation_split,
    )
    model.save(path / "model.h5")
    pickle.dump(history.history, open(path / "history.pickle", "wb"))
    try:
        plot_model(path)
    except ImportError:
        pass


if __name__ == "__main__":
    main(Path("../../data/track1"), epochs=5, validation_split=0.2, batch_size=None)
