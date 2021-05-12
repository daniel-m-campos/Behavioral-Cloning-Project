import csv
import math
import os
import random
import shutil
from collections import deque
from itertools import islice
from pathlib import Path
from typing import Tuple, Generator, List, Any

import cv2
import numpy as np
from dataclasses import dataclass


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
) -> Generator[TrainingData, None, None]:
    side = side if side is not None else ""
    log_path = path / log
    with open(log_path, "r") as file:
        reader = csv.reader(file)
        for line in reader:
            for img in line[:3]:
                img_path = Path(img)
                img = cv2.imread(str(path / img_dir / img_path.name))
                if img is not None and side in img_path.name:
                    yield TrainingData(
                        name=img_path.name,
                        steering_angle=float(line[-4]),
                        throttle=float(line[-3]),
                        brake=float(line[-2]),
                        speed=float(line[-1]),
                        image=img,
                    )


def shift_non_center_angles(
    training_data: Generator[TrainingData, None, None], shift: float
) -> Generator[TrainingData, None, None]:
    for train_data in training_data:
        if "right" in train_data.name:
            train_data.steering_angle -= shift
        elif "left" in train_data.name:
            train_data.steering_angle += shift
        yield train_data


def convert(
    training_data: List[TrainingData],
) -> Tuple[np.array, np.array]:
    x_train = np.array([x.image for x in training_data])
    y_train = np.array([x.steering_angle for x in training_data])
    return x_train, y_train


def add_horizontally_flipped(x: np.array, y: np.array) -> Tuple[np.array, np.array]:
    flipped_x = x[:, :, ::-1, :]
    flipped_y = -y.copy()
    return np.concatenate([x, flipped_x]), np.concatenate([y, flipped_y])


def create_batch_generator(
    generator: Generator[TrainingData, None, None], n: int
) -> Generator[List[TrainingData], None, None]:
    if n is None:
        yield list(generator)
    while True:
        yield list(islice(generator, n))


def create_validation_splitter(
    data_generator: Generator[Any, None, None],
    validation_split: float,
) -> Tuple[Generator[Any, None, None], Generator[Any, None, None]]:
    queues = [deque(), deque()]

    def fill_queues():
        x = next(data_generator)
        if random.random() > validation_split:
            queues[0].append(x)
        else:
            queues[1].append(x)

    def iter_from_queue(q):
        while True:
            while not q:
                try:
                    fill_queues()
                except StopIteration:
                    return
            yield q.popleft()

    return iter_from_queue(queues[0]), iter_from_queue(queues[1])


def create_data_generator(
    path: Path, batch_size: int = None, side: str = None, shift: float = None
) -> Generator[Tuple[np.array, np.array], None, None]:
    data_generator = read(path, side=side)
    yield from create_array_generator(data_generator, batch_size, shift)


def create_array_generator(
    data_generator: Generator[TrainingData, None, None],
    batch_size: int = None,
    shift: float = None,
) -> Generator[Tuple[np.array, np.array], None, None]:
    if isinstance(shift, float):
        data_generator = shift_non_center_angles(data_generator, shift)
    data_generator = create_batch_generator(data_generator, batch_size)
    while True:
        x_train, y_train = convert(next(data_generator))
        if len(x_train) > 0:
            yield add_horizontally_flipped(x_train, y_train)
        else:
            raise StopIteration


def create_validation_generators(
    path: Path,
    validation_split: float,
    batch_size: int = None,
    side: str = None,
    shift: float = None,
) -> Tuple[
    Generator[Tuple[np.array, np.array], None, None],
    Generator[Tuple[np.array, np.array], None, None],
]:
    data_generator = read(path, side=side)
    trainer, validator = create_validation_splitter(data_generator, validation_split)
    return (
        create_array_generator(trainer, batch_size, shift),
        create_array_generator(validator, batch_size, shift),
    )
