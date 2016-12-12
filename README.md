# Behavioral Cloning

In this project, we used a car simulator to teach a Convolutional Neural Network to drive a car around a track. The Network is only passed images from a front facing camera and the normalized steering angle in which the vehicle is being turned.

Our approach to solve this problem was to use transfer learning methods to use a pre-trained network for base features and fine-tune a Fully-Connected Neural Network place on top of it.

Our results are very successful, we were able to constantly drive around the first track for over 4 hours (until stopped) with no issues at all. We are also able to drive through the path of the second track multiple times.

## Code usage

In this project, you will find 3 scripts and 2 files representing the final model.

Scripts:

* train.py: this file allows you to train a model base on the driving data collected from Udacity's Simulator.
* tweak.py: this file allows you to fine-tune the pre-trained model by connecting a pygame hook and listening to keyboard input. You will select the 'Autonomous' mode in the Udacity Simulator and will correct or tweak the agents driving while it is predicting it's steering angles around the track. This new data will be collected and the script will train over this data for 10 epochs set once a Ctrl-C signal is sent to it.
* drive.py: this file allows you to let the agent drive autonomously around the selected track.

## Model Architecture

### Architecture

We decided to use the VGG16 base layers to begin training this network. In specific, the VGG16 network contains 16 trainable layers. However, in our case, we only kept the bottom 13 convolutional layers and the corresponding max pooling layers. Additionally, we attached a Fully-Connected Neural Network; this network contained 5 total layers in an attempt to built a Fully-Connected architecture similar to what we read in this [NVIDIA paper](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf). That is we connected a layer with 1024 neurons to another layer with 128 neurons, to a 64, then a 16, with a readout layer of 1 neuron.

### Regularization

Being such a large network and having only 2 tracks to be able to train on, we had to aggresively add regularization methods to our network. First, the VGG16 network come with max pooling layers which progressively reduces the size of the representation in order to reduce the amount of parameters and computation in the network, this way helps in the prevention of overfitting. The base layers of the VGG16 network contains 5 blocks of 2-3 convolutions each. Each of these blocks has a max pooling layer at the end of its last convolution.

Additionally to the max pooling layers, we added a dropout layer after each of the dense layers on the top network. These dropout layers also work to reduce overfitting by randomly dropping a specified number of connections each forward pass, therefore forcing the network to learn important features in different neurons, and simultaneously preventing complex adaptations to the training data.

Finally, we pre-recorded validation data on a seperate folder and use this as an independent validation set. We developed an early stopping function that would be queried after each epoch. The function will basically stop training if it found that the validation loss was improving for the past 2 epochs.

## Training Strategy

### Feature Transfer

For our training, we first leverage the feature layers of the VGG16 network. The network we started with had it's weights previously trained on the 'ImageNet' data set.

First, we set all of the pre-trained VGG16 features layer so that they are not trainable. In other words, it's weight will not change. Then, we connect the top layer and train a single epoch of a subset of the data. This allows for the weights of the top layers to be initialized to sensible values that work with the features currently on the VGG layers.

Next, we proceed to make only the upper 2 blocks of the VGG16 network trainable. This is so that the base features learned with the 'ImageNet' training stay in place. These features are only the base and therefore will consist of simple shapes and line like circles, squares and so on. Enabling these upper 2 blocks for training is very important because we want the network to 'forget' about cats, planes and horses, and learn about roads, trees, mountain, lane-lines, etc. These new items will most likely be learned on the top 2 layers only and the whole network is therefore able to learn new things.

Finally, we train on the whole data for several epochs until the loss in the validation set stops improving.

### Data Collection

Collecting the data was one of the most important steps during this project. In fact, one of the most important things we learned is that "is all about the data". This couldn't have been more true.

One of the main issues we had to collect the data is that the keyboard doesn't do justice for the way steering wheels work. For example, if we pressed the left key, the simulator would receive a left signal and the magnitude of this signal would increase exponentially. So, the longer you pressed the key, the more the turning would add up to be. In real life, on is able to keep a constant value on the steering wheel when turning which makes for a much easier training environment. Additonally, when we release the key, the signal goes immediatelly to a flat 0. This is obviously, not only challenging, but just not the way cars work in real life.

To deal with these issues, we design a training and driving strategy that consisted of the following:

1. Drive around the tracks with very rapid and constant presses, this way avoiding peaks signals.
2. Drive in the reverse of the tracks the same way as (1).
3. Drive again around the first track but this time using selective recording. We disable data recording, go next to the right lane line, enable recording, drive into the middle of the lane, disable recoding.
4. Drive as in (3) for every terrain patch you can encounter. Dirt, grass, yellow lines, white lines.
5. Repeat (3) and (4) but this time for the left lane line.
6. Do the same but for the second track.
7. Do the same but for the reverse of both tracks.

Additionally, after using this driving strategy and training the base model, we developed a tweak script that would intercept the trained agent's predictions and merge them with a user input. This script is very useful for specific corrections on a working model and it basically improves on behavior that might not be all that comfortable for the passengers inside. For example, the script uses the same main code as in train.py, but it also listens to key presses from the user. If the left or right keys are pressed, a very small value would be added to the predicted value so as to 'fix' an already good prediction. These image and adjusted prediction values would be stored on a numpy array and use for later for training. When the user feels like got enough and useful corrections on the current driving behavior, a simple Ctrl-C will kill the server sending the messages to the simulator and engage in training. The data is passed through with a very small learning rate so as to not damage or overfit the data, and for only a few epochs.

### Hyper-parameter Tunning


## Results
