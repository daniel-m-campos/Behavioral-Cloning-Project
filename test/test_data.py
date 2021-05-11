import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import pytest

from behavior_cloning import data

RESOURCE_PATH = path = f"{os.path.dirname(__file__)}/resources"


@pytest.fixture
def resource_path():
    return Path(RESOURCE_PATH)


@pytest.mark.skip
@pytest.mark.parametrize(
    "old_path, max_images", [("/Users/danielmcampos/Downloads/data", 10)]
)
def test_copy(old_path, resource_path, max_images):
    data.copy(Path(old_path), resource_path, max_images)


@pytest.fixture
def training_data(resource_path):
    return data.read(resource_path)


def test_read(training_data):
    assert len(training_data) == 36
    assert training_data[0].name == "center_2021_05_06_12_33_46_793.jpg"
    assert training_data[0].steering_angle == 0
    assert training_data[0].throttle == 0
    assert training_data[0].brake == 0
    assert training_data[0].speed == 7.997358e-05
    assert training_data[0].image.shape == (160, 320, 3)


def test_convert(training_data):
    x, y = data.convert(training_data)
    assert x.shape == (36, 160, 320, 3)
    assert y.shape == (36,)


def test_add_horizontally_flipped(training_data):
    x, y = data.convert(training_data)
    new_x, new_y = data.add_horizontally_flipped(x, y)
    assert new_x.shape[0] == 2 * x.shape[0]
    assert new_y.shape[0] == 2 * y.shape[0]
    assert np.allclose(new_x[: x.shape[0]], x)
    assert np.allclose(new_x[x.shape[0]], x[0, :, ::-1, :])
    assert new_y[x.shape[0]] == -y[0]
    assert np.allclose(new_x[-1], x[-1, :, ::-1, :])
    assert new_y[-1] == -y[-1]
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 1)
        axes[0].imshow(x[0])
        axes[1].imshow(new_x[x.shape[0]])
        plt.show()
    except ImportError:
        pass


def test_get_timestamp():
    timestamp = "2021_05_06_12_33_46_793"
    train_data = data.TrainingData(f"center_{timestamp}.jpg", 0, 0, 0, 0, None)
    assert train_data.get_timestamp() == timestamp


def test_shift_non_center_angles(training_data):
    shift = 0.5
    data.shift_non_center_angles(training_data, shift)
    img_by_timestamp = defaultdict(dict)
    for x in training_data:
        img_by_timestamp[x.get_timestamp()][x.name] = x
    for timestamp, timestamp_images in img_by_timestamp.items():
        center_angle = timestamp_images[f"center_{timestamp}.jpg"].steering_angle
        left_angle = timestamp_images[f"left_{timestamp}.jpg"].steering_angle
        right_angle = timestamp_images[f"right_{timestamp}.jpg"].steering_angle
        assert left_angle == pytest.approx(center_angle + shift, 1e-6)
        assert right_angle == pytest.approx(center_angle - shift, 1e-6)
