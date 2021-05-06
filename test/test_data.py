import os
from pathlib import Path

import pytest

from behavior_cloning import data

RESOURCE_PATH = path = f"{os.path.dirname(__file__)}/resources"


@pytest.fixture()
def resource_path():
    return Path(RESOURCE_PATH)


@pytest.mark.skip
@pytest.mark.parametrize(
    "old_path, max_images", [("/Users/danielmcampos/Downloads/data", 10)]
)
def test_copy(old_path, resource_path, max_images):
    data.copy(Path(old_path), resource_path, max_images)


def test_read(resource_path):
    training_data = data.read(resource_path)
    assert len(training_data) == 36
    assert training_data[0].name == "center_2021_05_06_12_33_46_793.jpg"
    assert training_data[0].steering_angle == 0
    assert training_data[0].throttle == 0
    assert training_data[0].brake == 0
    assert training_data[0].speed == 7.997358e-05
    assert training_data[0].image.shape == (160, 320, 3)
