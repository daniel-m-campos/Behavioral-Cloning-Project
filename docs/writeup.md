# Behavioral Cloning Project

The goals / steps of this project are the following:

* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

[//]: # (Image References)

[image1]: ../img/final_history.png "Final Model History"

[image2]: ../img/history_dropout.png "DropoutHistory"

[image3]: ../img/history_nodropout.png "No Dropout History"

[image4]: ../img/model.png "Model Visualization"

[image5]: ../img/center_example.jpg "Center Image"

[image6]: ../img/flipped_example.png "Flipped Image"

---

## Rubric Points

### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:

* `model.py` containing the script to create and train the model
* `drive.py` for driving the car in autonomous mode
* `model.h5` containing a trained convolution neural network
* `writeup.md` detailing project development

#### 2. Submission includes functional code

Using the Udacity simulator, and my `drive.py` file, the car can be driven autonomously
around the track by executing:

```bash
python drive.py data/track1/model.h5
```

#### 3. Submission code is usable and readable

* The `data.py` file contains the code for reading the image data and creating training
  and validating data generators.
* The `model.py` file contains the code to create and train the convolution neural
  network.
* The `__main__.py` file reads, trains, and saves the model.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model replicates the architecture developed by Nvidia with the addition of Lambda
normalizing layer followed by a cropping layer.

#### 2. Attempts to reduce over-fitting in the model

The model was trained and validated using a train and validation split of 30%. Further,
I tested the model by running it through the simulator and ensuring that the vehicle
could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually. I used 5
epochs even though it appears the validation loss is lowest after 2 epochs as 5 epochs
drove better on the track.

![alt text][image1]

#### 4. Appropriate training data

I generated training data by completing 3 laps around track. I added and subtracted a
shift of 0.125 to the left and right images. Lastly, I doubled the data set by
performing a left-right flip to remove the left turning bias from the data.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy was to start small and simple and work my way up to the full Nvidia
self-driving car architecture.

My first model had one normalizing layer followed by one dense layer and performed
terribly but demonstrated that the training, saving, and driving code was working. From
there I tried adding some convolutional layers which improved MSE and also performed
better on track tests. After this, I added a cropping layer and confirmed it performed
better on the track again.

After adding the cropping layer, I implemented Nvidia's architecture. The kernel and
stride parameters changed from in the final two Conv2D layers. This model was able to
navigate the entire track without incident.

I tested the model on track 2, but it failed immediately. I detoured into attempting to
fit track 2 with a similar training data and model approach but ran in to issues with
high memory use. I solved the memory issues by implementing train and validation data
generators. This consumed a lot of time, so I stopped to finish the project within the
time I had allocated.

Lastly, I tried to introduce dropout layers to prevent over-fitting, but the performance
on the track was always worse regardless of the dropout approach I took. Ultimately, I
settled on my initial architecture that I work on track 1.

#### With Dropout

Validation performance degraded with epoch using dropout layers.
![alt text][image2]

#### Without Dropout

Validation performance best without dropout after 2 epochs.
![alt text][image3]

#### 2. Final Model Architecture

![alt text][image4]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I recorded 3 laps on track one using center lane
driving. Here is an example image of center lane driving:

![alt text][image5]

Since I had 3 laps, I found that I didn't need to record recovery data since my data had
sufficient recovery behavior already.

Then I repeated this process on track 2 in order to get more data points, but didn't use
it in the end.

To augment the data sat, I also flipped images and angles thinking that this would
prevent a left turning bias. For example, here is an image that has then been flipped:

![alt text][image6]

After the collection process, I had 12,234x2 track 1 and 22,482x2 track 2 data points.
Lastly, I randomly shuffled the data set and put 30% of the data into a validation set.

## Conculsion

The model is able to do laps around track 1 but not track 2.

[Watch Track 1 Autonomous Driving](https://www.youtube.com/watch?v=Ug31MWvoORA)