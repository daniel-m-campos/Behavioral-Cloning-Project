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
