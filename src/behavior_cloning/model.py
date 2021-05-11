import numpy as np
from tensorflow import keras
from tensorflow.keras.layers import Flatten, Dense, Input
from tensorflow.keras.models import Sequential


def create() -> keras.Model:
    model = Sequential()
    model.add(Input(shape=(160, 320, 3)))
    model.add(Flatten())
    model.add(Dense(1))
    model.compile(loss="mse", optimizer="adam")
    return model


def train(x: np.array, y: np.array) -> keras.Model:
    model = create()
    model.fit(x, y, epochs=10, validation_split=0.2, shuffle=True)
    return model
