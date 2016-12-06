import tensorflow as tf
import pandas as pd
import numpy as np

from keras.models import Sequential, Model
from keras.layers import Dropout, Activation
from keras.layers import Input, Flatten, Dense
from keras.layers import Convolution2D, MaxPooling2D
from keras.wrappers.scikit_learn import KerasRegressor

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from scipy import misc


flags = tf.app.flags
FLAGS = flags.FLAGS

# command line flags
flags.DEFINE_integer('epochs', 5, 'The number of epochs.')
flags.DEFINE_integer('batch_size', 128, 'The batch size.')


def img_paths_to_img_array(image_paths):
    all_imgs = [misc.imresize(misc.imread(imp), (50,100)) for imp in image_paths]
    # all_imgs = [misc.imread(imp) for imp in image_paths]
    return np.array(all_imgs, dtype='float')

def create_model():

        model = Sequential()

        model.add(Convolution2D(32, 3, 3,
                                border_mode='valid',
                                input_shape=(50,100,3)))
                                #input_shape=(160,320,3)))

        model.add(Convolution2D(32, 3, 3))

        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Dropout(0.50))

        model.add(Flatten())
        model.add(Dense(3072, name='input'))
        model.add(Dropout(0.2))

        model.add(Dense(1536, name='hidden1'))
        model.add(Dropout(0.2))

        model.add(Dense(1, init='normal', name='output'))

        model.compile(loss='mean_squared_error', optimizer='adam')
        return model

def main(_):

    # fix random seed for reproducibility
    seed = 123
    np.random.seed(seed)

    # read the driving log
    with open('data/driving_log.csv', 'rb') as f:
        log_data = pd.read_csv(f, header=None, names=['center', 'left',
                                                      'right', 'angle',
                                                      'throttle', 'break',
                                                      'speed'])
    # extract the features and labels
    X_train = img_paths_to_img_array(log_data['center'])
    y_train = np.array(log_data['angle'].tolist())

    # normalize the images
    X_train /= 255.0
    X_train -= 0.5

    # split into training and validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.25, random_state=seed)

    # evaluate model with standardized dataset
    # regressor = KerasRegressor(build_fn=create_model,
    #                            nb_epoch=FLAGS.epochs,
    #                            batch_size=FLAGS.batch_size,
    #                            verbose=1)

    model = create_model()
    model.fit(X_train, y_train,
              batch_size=FLAGS.batch_size, nb_epoch=FLAGS.epochs,
              verbose=1, validation_data=(X_val, y_val))

    # kfold = KFold(n_splits=2, random_state=seed)
    # results = cross_val_score(regressor, X_train, y_train, cv=kfold)
    # print("Results: %.2f (%.2f) MSE" % (results.mean(), results.std()))

    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    model.save_weights('model.h5')

# parses flags and calls the `main` function above
if __name__ == '__main__':
    tf.app.run()
