import numpy as np
from tensorflow import keras
from tensorflow.keras.layers import (
    Flatten,
    Lambda,
    Dense,
    Cropping2D,
    Convolution2D,
    Input,
)
from tensorflow.keras.models import Sequential


def create() -> keras.Model:
    model = Sequential()
    model.add(Input(shape=(160, 320, 3)))
    model.add(Lambda(lambda x: (x / 255.0) - 0.5))
    model.add(Cropping2D(cropping=((70, 25), (0, 0))))
    model.add(Convolution2D(6, 5, 5, activation="relu"))
    model.add(Convolution2D(6, 5, 5, activation="relu"))
    model.add(Flatten())
    model.add(Dense(120))
    model.add(Dense(84))
    model.add(Dense(1))
    model.compile(loss="mse", optimizer="adam")
    return model


def train(
    x: np.array, y: np.array, epochs=10, validation_split=0.2, shuffle=True
) -> keras.Model:
    model = create()
    model.fit(x, y, epochs=epochs, validation_split=validation_split, shuffle=shuffle)
    return model
