from collections import defaultdict
from pathlib import Path

import numpy as np
import pytest

from behavior_cloning import data


@pytest.fixture
def resource_path():
    return Path(__file__).parent / "resources"


@pytest.mark.skip
@pytest.mark.parametrize(
    "old_path, max_images", [("/Users/danielmcampos/Downloads/data", 10)]
)
def test_copy(old_path, resource_path, max_images):
    data.copy(Path(old_path), resource_path, max_images)


@pytest.fixture
def training_data(resource_path):
    return list(data.read(resource_path))


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
        import cv2

        fig, axes = plt.subplots(2, 1)
        bgr2rgb = lambda img: cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        axes[0].imshow(bgr2rgb(x[0]))
        axes[0].set_title("Original")
        axes[1].imshow(bgr2rgb(new_x[x.shape[0]]))
        axes[1].set_title("Horizontally Flipped")
        for ax in axes:
            ax.axis("off")
        plt.show()
    except ImportError:
        pass


def test_get_timestamp():
    timestamp = "2021_05_06_12_33_46_793"
    train_data = data.TrainingData(f"center_{timestamp}.jpg", 0, 0, 0, 0, None)
    assert train_data.get_timestamp() == timestamp


def test_shift_non_center_angles(training_data):
    shift = 0.5
    training_data = data.shift_non_center_angles(training_data, shift)
    img_by_timestamp = defaultdict(dict)
    for x in training_data:
        img_by_timestamp[x.get_timestamp()][x.name] = x
    for timestamp, timestamp_images in img_by_timestamp.items():
        center_angle = timestamp_images[f"center_{timestamp}.jpg"].steering_angle
        left_angle = timestamp_images[f"left_{timestamp}.jpg"].steering_angle
        right_angle = timestamp_images[f"right_{timestamp}.jpg"].steering_angle
        assert left_angle == pytest.approx(center_angle + shift, 1e-6)
        assert right_angle == pytest.approx(center_angle - shift, 1e-6)


def test_create_data_generator(resource_path):
    data_generator = data.create_data_generator(resource_path, batch_size=6)
    batches = [sample for sample in data_generator]
    assert len(batches) == 6


def test_create_test_train_splitter(resource_path):
    trainer, validator = data.create_validation_generators(
        resource_path, 0.5, batch_size=5
    )
    validation_data = list(validator)
    train_data = list(trainer)
    assert len(train_data) > 0
    assert len(validation_data) > 0
