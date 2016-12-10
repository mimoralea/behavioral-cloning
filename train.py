import tensorflow as tf
import pandas as pd
import numpy as np

from keras.applications.inception_v3 import InceptionV3
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D

from keras.optimizers import Adam
from keras.models import Sequential, Model
from keras.layers import Dropout, Activation, Lambda
from keras.layers import Input, Flatten, Dense, ELU
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers.core import SpatialDropout2D
from keras.wrappers.scikit_learn import KerasRegressor
from keras.layers.normalization import BatchNormalization

import matplotlib.pyplot as plt

from scipy import misc
from skimage import color

flags = tf.app.flags
FLAGS = flags.FLAGS

# command line flags
flags.DEFINE_integer('features_epochs', 5,
                     'The number of epochs when training features.')
flags.DEFINE_integer('full_epochs', 100,
                     'The number of epochs when end-to-end training.')
flags.DEFINE_integer('tuning_epochs', 10,
                     'The number of epochs when tuning FCNN.')
flags.DEFINE_integer('batch_size', 128, 'The batch size.')
flags.DEFINE_integer('samples_per_epoch', 12800,
                     'The number of samples per epoch.')
flags.DEFINE_integer('val_size', 512, 'The batch size.')
flags.DEFINE_integer('img_h', 140, 'The image height.')
flags.DEFINE_integer('img_w', 200, 'The image width.')
flags.DEFINE_integer('img_c', 3, 'The number of channels.')

def img_pre_processing(img):
    # resize and cast to float
    img = misc.imresize(
        img, (FLAGS.img_h, FLAGS.img_w)).astype('float')

    #img = color.convert_colorspace(img, 'RGB', 'YUV')
    # normalize
    img /= 255.
    img -= 0.5
    img *= 2.
    return img

def img_paths_to_img_array(image_paths):
    all_imgs = [misc.imread(imp) for imp in image_paths]
    return np.array(all_imgs, dtype='float')

def save_model(model):
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    model.save_weights('model.h5')

def select_specific_set(iter_set):
    imgs, labs = [], []
    for _, row in iter_set:
        # extract the features and labels
        img = img_pre_processing(misc.imread(row['center']))
        lab = row['angle']

        # flip 50% of the time
        if np.random.choice([True, False]):
            img, lab = np.fliplr(img), -lab + 0.

        imgs.append(img)
        labs.append(lab)

    return np.array(imgs), np.array(labs)

def generate_batch(log_data):
    while True:
        """
        imgs1, labs1 = select_specific_set(
            log_data[log_data['angle'] > 0.1].sample(
                int(FLAGS.batch_size/4)).iterrows())
        imgs2, labs2 = select_specific_set(
            log_data[log_data['angle'] < -0.1].sample(
                int(FLAGS.batch_size/4)).iterrows())
        imgs3, labs3 = select_specific_set(log_data.sample(
            FLAGS.batch_size - len(imgs1) - len(imgs2)).iterrows())
        imgs = np.concatenate((imgs1, imgs2, imgs3), axis=0)
        labs = np.concatenate((labs1, labs2, labs3), axis=0)
        """
        imgs, labs = select_specific_set(log_data.sample(
            FLAGS.batch_size).iterrows())
        yield np.array(imgs), np.array(labs)

def main(_):

    # fix random seed for reproducibility
    np.random.seed(123)

    # read the driving log
    with open('data/driving_log.csv', 'rb') as f:
        log_data = pd.read_csv(
            f, header=None,
            names=['center', 'left', 'right', 'angle',
                   'throttle', 'break', 'speed'])

    # get a small set to use for validation
    X_val, y_val = select_specific_set(
        log_data.sample(FLAGS.val_size).iterrows())
    
    # create and train the model
    input_shape = (FLAGS.img_h, FLAGS.img_w, FLAGS.img_c)
    input_tensor = Input(shape=input_shape)

    # get the VGG16 network
    base_model = VGG16(input_tensor=input_tensor,
                             weights='imagenet',
                             include_top=False)

    # add a global spatial average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)

    # add the fully-connected
    # layer similar to the NVIDIA paper
    x = Dense(1024, activation='elu')(x)
    x = Dropout(0.4)(x)
    x = Dense(512, activation='elu')(x)
    x = Dropout(0.3)(x)
    x = Dense(128, activation='elu')(x)
    x = Dropout(0.2)(x)
    x = Dense(32, activation='elu')(x)
    x = Dropout(0.1)(x)
    predictions = Dense(1, init='zero')(x)

    # creatte the full model
    model = Model(input=base_model.input, output=predictions)

    # freeze all convolutional layers to initialize the top layers
    for layer in base_model.layers:
        layer.trainable = False
    
    # train the model to prepare all weights
    opt = Adam(lr=1e-04, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.5)
    model.compile(optimizer=opt, loss='mse')

    history = model.fit_generator(
        generate_batch(log_data),
        samples_per_epoch=FLAGS.samples_per_epoch,
        validation_data=(X_val, y_val),
        nb_epoch=FLAGS.features_epochs,
        verbose=1)

    # print all layers
    for i, layer in enumerate(base_model.layers):
        print(i, layer.name)

    # for VGG we choose to include the
    # top 2 blocks in training
    for layer in model.layers[:11]:
       layer.trainable = False
    for layer in model.layers[11:]:
       layer.trainable = True

    # recompile and train with a finer learning rate
    opt = Adam(lr=1e-04, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0)
    model.compile(optimizer=opt, loss='mse')

    history = model.fit_generator(
        generate_batch(log_data),
        samples_per_epoch=FLAGS.samples_per_epoch,
        validation_data=(X_val, y_val),
        nb_epoch=FLAGS.full_epochs,
        verbose=1)

    # fine-tune top layer only
    # freeze all convolutional layers
    for layer in base_model.layers:
        layer.trainable = False
    
    # recompile and train once more
    opt = Adam(lr=1e-06, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0)
    model.compile(optimizer=opt, loss='mse')

    history = model.fit_generator(
        generate_batch(log_data),
        samples_per_epoch=FLAGS.samples_per_epoch,
        validation_data=(X_val, y_val),
        nb_epoch=FLAGS.tuning_epochs,
        verbose=1)

    # save model to disk
    save_model(model)

# parses flags and calls the `main` function above
if __name__ == '__main__':
    tf.app.run()
