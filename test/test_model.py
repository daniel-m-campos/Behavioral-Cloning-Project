from pathlib import Path

import pytest
from tensorflow.keras.models import load_model

from behavior_cloning import data
from behavior_cloning import model as mdl


@pytest.fixture
def resource_path() -> Path:
    return Path(__file__).parent / "resources"


def test_create():
    model = mdl.create()
    print(model.summary())


def test_train(resource_path):
    model = mdl.train(*data.convert(data.read(resource_path)), epochs=0)
    assert len(model.layers[-1].get_weights()) > 0


def test_predict(resource_path):
    training_data = data.read(resource_path)
    model = load_model(resource_path / "model.h5")
    for i, image in enumerate(t.image for t in training_data):
        y = float(model.predict(image.reshape((-1, 160, 320, 3))))
        print(f"#{i}, Prediction =", y)
