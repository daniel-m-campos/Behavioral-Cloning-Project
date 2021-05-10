from pathlib import Path

import pytest
from tensorflow.keras.models import load_model

from behavior_cloning import data


@pytest.fixture
def resource_path() -> Path:
    return Path(__file__).parent / "resources"


def test_predict(resource_path):
    training_data = data.read(resource_path)
    model = load_model(resource_path / "model.h5")
    for i, image in enumerate(t.image for t in training_data):
        y = float(model.predict(image.reshape((-1, 160, 320, 3))))
        print(f"#{i}, Prediction =", y)
