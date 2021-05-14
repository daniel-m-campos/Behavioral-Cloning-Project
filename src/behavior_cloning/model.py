from typing import Tuple, Generator

import numpy as np
from tensorflow import keras
from tensorflow.keras.layers import (
    Dropout,
    Flatten,
    Lambda,
    Dense,
    Cropping2D,
    Convolution2D,
    Input,
)
from tensorflow.keras.models import Sequential


def create() -> keras.Model:
    """
    Create the Nvidia CNN network as described in:
        https://developer.nvidia.com/blog/deep-learning-self-driving-cars/
    """
    model = Sequential()
    model.add(Input(shape=(160, 320, 3)))
    model.add(Lambda(lambda x: (x / 255.0) - 0.5))
    model.add(Cropping2D(cropping=((70, 25), (0, 0))))
    model.add(Convolution2D(24, 5, 2, activation="relu"))
    model.add(Convolution2D(36, 5, 2, activation="relu"))
    model.add(Convolution2D(48, 5, 2, activation="relu"))
    model.add(Convolution2D(64, 3, 1, activation="relu"))
    model.add(Convolution2D(64, 3, 1, activation="relu"))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))
    model.compile(loss="mse", optimizer="adam")
    return model


def train(
    train_data_generator: Generator[Tuple[np.array, np.array], None, None],
    validation_data_generator: Generator[Tuple[np.array, np.array], None, None] = None,
    epochs=1,
    shuffle=True,
    steps_per_epoch=None,
    validation_steps=None,
    validation_split=None,
) -> Tuple[keras.Model, keras.callbacks.History]:
    x, y = train_data_generator, None
    if isinstance(validation_split, float):
        x, y = next(train_data_generator)
        validation_data_generator = None
        steps_per_epoch = None
        validation_steps = None
    model = create()
    history = model.fit(
        x=x,
        y=y,
        epochs=epochs,
        shuffle=shuffle,
        validation_split=validation_split,
        validation_data=validation_data_generator,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
    )
    return model, history
