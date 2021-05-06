import csv
import math
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import List

import cv2
import numpy as np


def copy(old_path: Path, new_path: Path, log="driving_log.csv", img_dir="IMG",
         max_lines=math.inf):
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


def read(path: Path, log="driving_log.csv", img_dir="IMG") -> List[TrainingData]:
    log_path = path / log
    training_data = []
    with open(log_path, "r") as file:
        reader = csv.reader(file)
        for line in reader:
            for img in line[:3]:
                img_path = Path(img)
                img = cv2.imread(str(path / img_dir / img_path.name))
                if img is not None:
                    training_data.append(
                        TrainingData(
                            name=img_path.name,
                            steering_angle=float(line[-4]),
                            throttle=float(line[-3]),
                            brake=float(line[-2]),
                            speed=float(line[-1]),
                            image=img)
                    )
    return training_data
