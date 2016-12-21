# Self-Driving Nanodegree

## Project 3: Behavioral Cloning

Miguel Morales | [@mimoralea](https://twitter.com/mimoralea) | [mimoralea@gmail.com](mailto:mimoralea@gmail.com)

In this project, we used a car simulator to teach a Convolutional Neural Network to drive a car around a track. The Network is only passed images from a front facing camera and the normalized steering angle in which the vehicle is being turned.

Our approach to solve this problem was to use transfer learning methods to use a pre-trained network for base features and fine-tune a Fully-Connected Neural Network place on top of it.

Our results are very successful, we were able to constantly drive around the first track for over 4 hours (until stopped) with no issues at all. We are also able to drive through the path of the second track multiple times.

![Behavioral Cloning][intro]
![Behavioral Cloning][intro2]

## Code usage

```
(keras) [mimoralea@hash behavioral-cloning]$ tree . -L 2
.
|-- README.md
|-- data
|   |-- train
|   `-- validation
|-- drive.py
|-- drive_old.py
|-- imgs
|   |-- becloning.gif
|   |-- becloning2.gif
|   |-- block5_conv1_filters_8x8.png
|   |-- block5_conv2_filters_8x8.png
|   |-- block5_conv3_filters_8x8.png
|   |-- cropped.png
|   |-- dropout.jpeg
|   |-- vgg16.png
|   `-- vgg16_filters_overview.jpg
|-- model.h5
|-- model.json
|-- model_old.h5
|-- model_old.json
|-- packages.txt
|-- plotweights.py
|-- simulator
|   |-- Default\ Linux\ desktop\ Universal.x86
|   |-- Default\ Linux\ desktop\ Universal.x86_64
|   |-- Default\ Linux\ desktop\ Universal_Data
|   `-- simulator-linux.zip
|-- train.py
`-- tweak.py
```

In this project, you will find 5 important files. 3 scripts to train, fine-tune and test the models, and 2 files representing the model architecture and weights.

Scripts:

* train.py: this file allows you to train a model base on the driving data collected from Udacity's Simulator.
* tweak.py: this file allows you to fine-tune the pre-trained model by connecting a pygame hook and listening to keyboard input. You will select the 'Autonomous' mode in the Udacity Simulator and will correct or tweak the agents driving while it is predicting it's steering angles around the track. This new data will be collected and the script will train over this data for 10 epochs set once a Ctrl-C signal is sent to it.
* drive.py: this file allows you to let the agent drive autonomously around the selected track.

Model files:

* model.json: represents the model architecture of the Convolutional Neural Network.
* model.h5: represents the trained weights of the model.

## Model Architecture

### Architecture

We decided to use the VGG16 base layers to begin training this network. In specific, the VGG16 network originally contains 16 trainable layers. However, in our case, we only kept the bottom 13 convolutional layers and the corresponding max pooling layers. Additionally, we attached a Fully-Connected Neural Network; this network contained 5 total layers in an attempt to built a Fully-Connected architecture similar to what we read in this [NVIDIA paper](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf). That is we connected a layer with 1024 neurons to another layer with 128 neurons, to a 64, then a 16, with a readout layer of 1 neuron.

![VGG16 Macro Architecture][vgg16]

In the image above (credit to [Davi Frossard](https://www.cs.toronto.edu/~frossard/post/vgg16/)), we can visualize the original architecture of the VGG16 network. On our solution, we removed the blue and yellow layers and replace them with an approximation of the Fully-Connected layers of the NVIDIA architecture:

`512 -> elu -> dropout -> 256 -> elu  -> dropout -> 64 -> elu  -> dropout -> 1`

### Regularization

Being such a large network and having only 2 tracks to be able to train on, we had to aggresively add regularization methods to our network. First, the VGG16 network come with max pooling layers which progressively reduces the size of the representation in order to reduce the amount of parameters and computation in the network, this way helps in the prevention of overfitting. The base layers of the VGG16 network contains 5 blocks of 2-3 convolutions each. Each of these blocks has a max pooling layer at the end of its last convolution.

Additionally to the max pooling layers, we added a dropout layer after each of the dense layers on the top network. These dropout layers also work to reduce overfitting by randomly dropping a specified number of connections each forward pass, therefore forcing the network to learn important features in different neurons, and simultaneously preventing complex adaptations to the training data.

![Dropout][dropout]

In the image above (credit to [Leonardo Araujo](https://leonardoaraujosantos.gitbooks.io/artificial-inteligence/content/dropout_layer.html) we can see a depiction of dropout layers. We added dropout layers after each of the dense layers.

Finally, we pre-recorded validation data on a seperate folder and use this as an independent validation set. We developed an early stopping function that would be queried after each epoch. The function will basically stop training if it found that the validation loss was improving for the past 2 epochs.

## Training Strategy

### Feature Transfer

For our training, we first leverage the feature layers of the VGG16 network. The network we started with had it's weights previously trained on the 'ImageNet' data set.

First, we set all of the pre-trained VGG16 features layer so that they are not trainable. In other words, it's weight will not change. Then, we connect the top layer and train a single epoch of a subset of the data. This allows for the weights of the top layers to be initialized to sensible values that work with the features currently on the VGG layers.

![Conv Weights][vggweights]

In the image above (credit to [François Chollet](https://blog.keras.io/how-convolutional-neural-networks-see-the-world.html)), we can see what our original network sees the world. The important thing to remember, for example, is that the 4th and 5th conv blocks got Fine-tuned with our new data set. So, all of the features learned in the first 3 convolutions will remain this way on our final model.

Next, we proceed to make only the upper 2 blocks of the VGG16 network trainable. This is so that the base features learned with the 'ImageNet' training stay in place. These features are only the base and therefore will consist of simple shapes and line like circles, squares and so on. Enabling these upper 2 blocks for training is very important because we want the network to 'forget' about cats, planes and horses, and learn about roads, trees, mountain, lane-lines, etc. These new items will most likely be learned on the top 2 layers only and the whole network is therefore able to learn new things.

Finally, we train on the whole data for several epochs until the loss in the validation set stops improving.

![Block 5 Conv 1][block5conv1]

![Block 5 Conv 2][block5conv2]

In the images above, we can see what some of the neurons on the 
last convolutional layers look like. As we can see from the image, 
the convolutions progresses the images become more and more elaborated. 
This is good and an indication that our Neural Network is able to 
learn from the most basic to the most complex representations and 
with this able to correctly predict when to turn and in what degree.

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

![Behavioral Cloning][intro]

![Behavioral Cloning][intro2]

The gifs above were created from the data collected performing the steps as above.

In addition to the steps above, after using this driving strategy and training the base
 model, we developed a tweak script that would intercept the trained agent's 
predictions and merge them with a user input. This script is very useful for 
specific corrections on a working model and it basically improves on behavior
 that might not be all that comfortable for the passengers inside. For example, 
the script uses the same main code as in train.py, but it also listens to key 
presses from the user. If the left or right keys are pressed, a very small 
value would be added to the predicted value so as to 'fix' an already good 
prediction. These image and adjusted prediction values would be stored on a
 numpy array and use for later for training. When the user feels like got 
enough and useful corrections on the current driving behavior, a simple 
Ctrl-C will kill the server sending the messages to the simulator and 
engage in training. The data is passed through with a very small learning 
rate so as to not damage or overfit the data, and for only a few epochs.

### Training, Validation & Testing

As mentioned in the section above, we collected separate and independent 
Training and Validation data sets. This ensures the accuracy of the validation 
set loss, and helps with preventing overfitting.

For testing we use the simulators, this is really the only accurate way of 
determining whether the agent is capable or not of driving around each of
 the tracks. Creating a test set ourselves would not be of great advantage 
in this case. Mostly because there is no easily recognizable ground truth 
and the only thing we need to prove is whether the agent can drive on the 
tracks or not.

On the main track we left the agent driving for about 4 hours continuously
 and the agent was able to successfully drive around without any issues 
until user intervention.

### Hyper-parameter Tunning

Since we used an Adam optimizer, most of the hyper parameter tuning is 
done internally on it. However, we did select a learning rate that is high 
enough so to train fast, and not so high so that the agent would actually
 learn. Additionally, we use a decay rate of about 0.7, this way the 
initial learning rate will decrease and the model will get more precise 
as the epochs increased.

### Batch Generator

We developed a batch generator that would allow us to yield unlimited combinations
of images in batches as required. In this batch generation function we added a couple
of special features. First, the image pre-processing function would be called on
on the images to be added to the batch. This reduced the amount of computation
and additionally improve training time because these data augmentation
strategies would run on the CPU while the training process would run on the GPU.
The pre-processing of the image included image resize to a 100x200 ratio and then
cropping the image to 40 rows below the horizon as shown in the image below:

![Cropping Horizon][crophor]

Then, we do image normalization with values laying in between -1 and 1.
We tried using image convertion to YUV space instead of RGB,
but perhaps since we used transfer learning and the 'ImageNet'
weights had been acquired with RGB images, it
was better not to do so.

After the image was resized and normalized, it was horizontally flipped
with a 50% chance. If the image was flipped, then the corresponding labels
was multiplied by -1 to look for the corresponding turning angle. Then the images
would be appended to a batch of 128 and passed to the training procedure.

## Results

An image is worth a thousand words, an video is worth... enjoy.

### Track 1

[![Alt text](https://img.youtube.com/vi/H3Ifr-j4vBU/0.jpg)](https://www.youtube.com/watch?v=H3Ifr-j4vBU)

### Track 2

[![Alt text](https://img.youtube.com/vi/aDfaWdODU_0/0.jpg)](https://www.youtube.com/watch?v=aDfaWdODU_0)


[intro]: ./imgs/becloning.gif "Behavioral Cloning"
[intro2]: ./imgs/becloning2.gif "Behavioral Cloning"
[vgg16]: ./imgs/vgg16.png "VGG16 Original Macro Architecture"
[dropout]: ./imgs/dropout.jpeg "Dropout"
[vggweights]: ./imgs/vgg16_filters_overview.jpg "VGG16 ImageNet Weights"
[block5conv1]: ./imgs/block5_conv1_filters_5x5.png "Final Model Weights Block 5 conv 1"
[block5conv2]: ./imgs/block5_conv2_filters_5x5.png "Final Model Weights Block 5 conv 2"
[crophor]: ./imgs/cropped.png "Cropped Horizon"