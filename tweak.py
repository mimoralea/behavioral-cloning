import argparse
import base64
import json
import pygame

import signal
import sys

import numpy as np
import socketio
import eventlet
import eventlet.wsgi
import time
from PIL import Image
from PIL import ImageOps
from flask import Flask, render_template
from io import BytesIO
from scipy import misc

from time import sleep
import threading
from keras.models import model_from_json
from train import img_pre_processing
from keras.optimizers import Adam

X_corr, y_corr = [], []
model = None
val = 0

sio = socketio.Server(async_mode='eventlet')
app = socketio.Middleware(sio)

pygame.init()
pygame.display.set_caption('')
screen = pygame.display.set_mode((200,60), pygame.DOUBLEBUF)


def send_control(steering_angle, throttle, nsamples=0):
    print('angle:', steering_angle, 'throttle:', throttle, 'samples:', nsamples)
    sio.emit('steer', data={
    'steering_angle': steering_angle.__str__(),
    'throttle': throttle.__str__()
    }, skip_sid=True)

@sio.on('telemetry')
def telemetry(sid, data):
    global val, X_corr, y_corr, screen

    imgString = data['image']
    image = Image.open(BytesIO(base64.b64decode(imgString)))
    image_array = np.asarray(image)

    img = img_pre_processing(image_array)
    img_batch = img[None, :, :, :].astype('float')

    steering_angle = model.predict(img_batch, batch_size=1)

    # calculate angle corrections
    key = pygame.key.get_pressed()
    if key[pygame.K_LEFT]:
        val -= 0.005
        val = np.max((val, -1))
    elif key[pygame.K_RIGHT]:
        val += 0.005
        val = np.min((val, 1))
    else:
        val = 0

    # calculate corrected angle and send control
    angle = steering_angle[0][0] + val

    # save corrections
    if val != 0:
        X_corr.append(img.astype('float'))
        y_corr.append([angle])
    
    # control car
    send_control(angle,
                 1. if abs(angle) < 0.2 else 0.3 if abs(angle) < 0.5 else -0.5,
                 len(X_corr))

    surface = pygame.surfarray.make_surface(np.flipud(np.rot90(img)))
    screen.blit(surface, (0, 0))
    pygame.display.flip()    

    # refresh keyboard state
    pygame.event.pump()


def signal_handler(signal, frame):
    global X_corr, y_corr, screen
    print('Found', len(X_corr), 'to train with')

    if len(X_corr) > 0:
        X_corr = np.array(X_corr, dtype='float')
        y_corr = np.array(y_corr, dtype='float')
        print('Training with', X_corr.shape[0], 'samples')

        # for VGG we choose to include the
        # top 2 blocks in training
        for layer in model.layers[:11]:
           layer.trainable = False
        for layer in model.layers[11:]:
           layer.trainable = True

        model.fit(X_corr, y_corr,
                  batch_size=128,
                  nb_epoch=3,
                  verbose=1)

        timestamp = str(int(time.time()))
        json_filename = 'model_' + timestamp + '.json'
        weights_filename = 'model_' + timestamp + '.h5'

        model_json = model.to_json()
        with open(json_filename, 'w') as json_file:
            json_file.write(model_json)
        model.save_weights(weights_filename)

        print('Model saved at: ', json_filename)
        print('Weights saved at: ', weights_filename)

    sys.exit(0)


@sio.on('connect')
def connect(sid, environ):
    print('connect ', sid)
    send_control(0, 0)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument('model', type=str,
    help='Path to model definition json. Model weights should be on the same path.')
    args = parser.parse_args()
    with open(args.model, 'r') as jfile:
        json_model = jfile.read()
        model = model_from_json(json_model)

    opt = Adam(lr=1e-05, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0)
    model.compile(optimizer=opt, loss='mse')

    weights_file = args.model.replace('json', 'h5')
    model.load_weights(weights_file)

    # deploy as an eventlet WSGI server
    signal.signal(signal.SIGINT, signal_handler)
    print('Press Ctrl+C when ready to train')
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
