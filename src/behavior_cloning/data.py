import csv
import math
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np


def copy(
    old_path: Path,
    new_path: Path,
    log="driving_log.csv",
    img_dir="IMG",
    max_lines=math.inf,
):
    dirs = os.listdir(old_path)
    assert log in dirs
    assert img_dir in dirs
    new_log = new_path / log
    shutil.copy(old_path / log, new_log)

    os.mkdir(new_path / img_dir)
    with open(new_log, "r") as file:
        reader = csv.reader(file)
        for count, line in enumerate(reader):
            for img in line[:3]:
                old_img = old_path / img_dir / Path(img).name
                new_img = new_path / img_dir / Path(img).name
                shutil.copy(old_img, new_img)
            if count > max_lines:
                break


@dataclass
class TrainingData:
    name: str
    steering_angle: float
    throttle: float
    brake: float
    speed: float
    image: np.array((160, 320, 3))

    def get_timestamp(self):
        return "_".join(self.name.replace(".jpg", "").split("_")[1:])


def read(
    path: Path, log="driving_log.csv", img_dir="IMG", side=None
) -> List[TrainingData]:
    side = side if side is not None else ""
    log_path = path / log
    training_data = []
    with open(log_path, "r") as file:
        reader = csv.reader(file)
        for line in reader:
            for img in line[:3]:
                img_path = Path(img)
                img = cv2.imread(str(path / img_dir / img_path.name))
                if img is not None and side in img_path.name:
                    training_data.append(
                        TrainingData(
                            name=img_path.name,
                            steering_angle=float(line[-4]),
                            throttle=float(line[-3]),
                            brake=float(line[-2]),
                            speed=float(line[-1]),
                            image=img,
                        )
                    )
    return training_data


def shift_non_center_angles(training_data: List[TrainingData], shift: float):
    for train_data in training_data:
        if "center" in train_data.name:
            continue
        elif "right" in train_data.name:
            train_data.steering_angle -= shift
        elif "left" in train_data.name:
            train_data.steering_angle += shift
        else:
            raise ValueError("Name should be either left, center, or right")


def convert(training_data: List[TrainingData]) -> Tuple[np.array, np.array]:
    x_train = np.array([x.image for x in training_data])
    y_train = np.array([x.steering_angle for x in training_data])
    return x_train, y_train


def add_horizontally_flipped(x: np.array, y: np.array) -> Tuple[np.array, np.array]:
    flipped_x = x[:, :, ::-1, :]
    flipped_y = -y.copy()
    return np.concatenate([x, flipped_x]), np.concatenate([y, flipped_y])


def get_training_data(path: Path, side=None, shift=None):
    training_data = read(path, side=side)
    if isinstance(shift, float):
        shift_non_center_angles(training_data, shift)
    return add_horizontally_flipped(*convert(training_data))
