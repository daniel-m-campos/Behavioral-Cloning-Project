# Behavioral Cloning Project [![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive) [![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

![](img/autonomous_driving.png)



## The Project

The goals / steps of this project are the following:

* Use the simulator to collect data of good driving behavior
* Design, train and validate a model that predicts a steering angle from image data
* Use the model to drive the vehicle autonomously around the first track in the
  simulator. The vehicle should remain on the road for an entire loop around the track.
* Summarize the results with a written report

### Installation

```bash
git clone https://github.com/daniel-m-campos/Behavioral-Cloning-Project.git
cd Behavioral-Cloning-Project
pip install . --no-deps
```

#### Dependencies

This project uses Python 3.6, Tensorflow 2.4, and Linux only. Dependencies
for `behavior_cloning`
package can be installed with `conda` or `pip`.

##### Conda

```bash
conda env create -f environment.yml
```

##### Pip

```bash
virtualenv venv
. venv/bin/activate
pip install -r requirements.txt
```

### Downloading Data

The data used for training and driving can be downloaded by:

```bash
cd Behavioral-Cloning-Project
bash download_data.sh
```

### Training

A model can be trained by specifying the simulator training directory:

```bash
python -m behavior_cloning data/track1
```

## Module Details

### `data.py`

The `data.py` module provides utilities for parsing the simulator training data using
generators where appropriate.

### `model.py`

This module provides a `create()` function for the model and a `train()` which creates
and trains the model given data generators.

### `__main__.py`

This module glues the `data.py` and `model.py` and provides a command line entry point.

### `drive.py`

Usage of `drive.py` requires you have saved the trained model as an h5 file,
i.e. `model.h5`. See
the [Keras documentation](https://keras.io/getting-started/faq/#how-can-i-save-a-keras-model)
for how to create this file using the following command:

```sh
model.save(filepath)
```

Once the model has been saved, it can be used with drive.py using this command:

```sh
python drive.py model.h5
```

The above command will load the trained model and use the model to make predictions on
individual images in real-time and send the predicted angle back to the server via a
websocket connection.

Note: There is known local system's setting issue with replacing "," with "." when using
drive.py. When this happens it can make predicted steering values clipped to max/min
values. If this occurs, a known fix for this is to add "export LANG=en_US.utf8" to the
bashrc file.

#### Saving a video of the autonomous agent

```sh
python drive.py model.h5 run1
```

The fourth argument, `run1`, is the directory in which to save the images seen by the
agent. If the directory already exists, it'll be overwritten.

```sh
ls run1

[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_424.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_451.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_477.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_528.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_573.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_618.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_697.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_723.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_749.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_817.jpg
...
```

The image file name is a timestamp of when the image was seen. This information is used
by `video.py` to create a chronological video of the agent driving.

### `video.py`

```sh
python video.py run1
```

Creates a video based on images found in the `run1` directory. The name of the video
will be the name of the directory followed by `'.mp4'`, so, in this case the video will
be `run1.mp4`.

Optionally, one can specify the FPS (frames per second) of the video:

```sh
python video.py run1 --fps 48
```

Will run the video at 48 FPS. The default FPS is 60.



