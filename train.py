import tensorflow as tf
import pandas as pd
import numpy as np

from keras.optimizers import Adam
from keras.models import Sequential, Model
from keras.layers import Dropout, Activation, Lambda
from keras.layers import Input, Flatten, Dense, ELU
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers.core import SpatialDropout2D
from keras.wrappers.scikit_learn import KerasRegressor
from keras.layers.normalization import BatchNormalization

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

import matplotlib.pyplot as plt

from scipy import misc
from skimage import color

flags = tf.app.flags
FLAGS = flags.FLAGS

# command line flags
flags.DEFINE_integer('epochs', 5, 'The number of epochs.')
flags.DEFINE_integer('batch_size', 256, 'The batch size.')
flags.DEFINE_integer('samples_per_epoch', 25600,
                     'The number of samples per epoch.')
flags.DEFINE_integer('val_size', 1000, 'The batch size.')
flags.DEFINE_integer('img_h', 66, 'The image height.')
flags.DEFINE_integer('img_w', 200, 'The image width.')
flags.DEFINE_integer('img_c', 3, 'The number of channels.')

def img_pre_processing(img):
    img = misc.imresize(img.astype('float'),
                        (FLAGS.img_h, FLAGS.img_w))
    img = color.convert_colorspace(img, 'RGB', 'YUV')
    return img

def img_paths_to_img_array(image_paths):
    all_imgs = [misc.imread(imp) for imp in image_paths]
    return np.array(all_imgs, dtype='float')

def create_model():

    input_shape = (FLAGS.img_h, FLAGS.img_w, FLAGS.img_c)

    model = Sequential()
    model.add(BatchNormalization(input_shape=input_shape))
    model.add(Convolution2D(24, 5, 5,
                            subsample=(2,2),
                            border_mode='same',
                            input_shape=input_shape,
                            name='conv1'))
    model.add(Convolution2D(36, 4, 4,
                            subsample=(2,2),
                            border_mode='same',
                            name='conv2'))
    model.add(Convolution2D(48, 3, 3,
                            subsample=(2,2),
                            border_mode='same',
                            name='conv3'))
    model.add(Convolution2D(64, 2, 2,
                            subsample=(1,1),
                            border_mode='same',
                            name='conv4'))
    model.add(Convolution2D(64, 2, 2,
                            subsample=(1,1),
                            border_mode='same',
                            name='conv5'))

    model.add(Flatten())
    model.add(Dense(1024, name='input'))
    model.add(ELU())
    model.add(Dropout(.5))

    model.add(Dense(512, name='hidden1'))
    model.add(ELU())
    model.add(Dropout(.4))

    model.add(Dense(128, name='hidden2'))
    model.add(ELU())
    model.add(Dropout(.3))

    model.add(Dense(32, name='hidden3'))
    model.add(ELU())
    model.add(Dropout(.2))

    model.add(Dense(1, init='zero', name='output'))

    opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.2)
    model.compile(optimizer=opt, loss='mean_squared_error')
    return model

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

def generate_batch(log_data, training=True):
    while training:
        imgs1, labs1 = select_specific_set(
            log_data[log_data['angle'] > 0].sample(
                int(FLAGS.batch_size/5)).iterrows())
        imgs2, labs2 = select_specific_set(
            log_data[log_data['angle'] < 0].sample(
                int(FLAGS.batch_size/5)).iterrows())
        imgs3, labs3 = select_specific_set(log_data.sample(
            FLAGS.batch_size - len(imgs1) - len(imgs2)).iterrows())
        imgs = np.concatenate((imgs1, imgs2, imgs3), axis=0)
        labs = np.concatenate((labs1, labs2, labs3), axis=0)
        yield np.array(imgs), np.array(labs)

    while not training:
        imgs, labs = select_specific_set(log_data.sample(
            FLAGS.val_size).iterrows())
        yield np.array(imgs), np.array(labs)

def main(_):

    # fix random seed for reproducibility
    seed = 123
    np.random.seed(seed)

    # read the driving log
    with open('data/driving_log.csv', 'rb') as f:
        log_data = pd.read_csv(
            f, header=None,
            names=['center', 'left', 'right', 'angle',
                   'throttle', 'break', 'speed'])

    # create and train the model
    model = create_model()
    history = model.fit_generator(
        generate_batch(log_data),
        samples_per_epoch=FLAGS.samples_per_epoch,
        validation_data=generate_batch(log_data, False),
        nb_val_samples=FLAGS.val_size,
        nb_epoch=FLAGS.epochs,
        verbose=1)

    # save model to disk
    save_model(model)

# parses flags and calls the `main` function above
if __name__ == '__main__':
    tf.app.run()
