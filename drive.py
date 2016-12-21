import argparse
import base64
import json

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

from keras.models import model_from_json
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array
from train import img_pre_processing

sio = socketio.Server()
app = Flask(__name__)
model = None

def send_control(steering_angle, throttle):
    print('angle:', steering_angle, 'throttle:', throttle)
    sio.emit("steer", data={
    'steering_angle': steering_angle.__str__(),
    'throttle': throttle.__str__()
    }, skip_sid=True)

@sio.on('telemetry')
def telemetry(sid, data):
    # The current steering angle of the car
    # steering_angle = data["steering_angle"]
    # The current throttle of the car
    # throttle = data["throttle"]
    # The current speed of the car
    # speed = data["speed"]
    # The current image from the center camera of the car
    imgString = data["image"]
    image = Image.open(BytesIO(base64.b64decode(imgString)))
    image_array = np.asarray(image)

    img = img_pre_processing(image_array)
    img_batch = img[None, :, :, :].astype('float')

    steering_angle = model.predict(img_batch, batch_size=1)

    angle = steering_angle[0][0]
    send_control(
        angle,
        1. if abs(angle) < 0.2 else 0.3 if abs(angle) < 0.5 else -0.5)



@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send_control(0, 0)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument('model', type=str,
    help='Path to model definition json. Model weights should be on the same path.')
    args = parser.parse_args()
    with open(args.model, 'r') as jfile:
        json_model = jfile.read()
        model = model_from_json(json_model)

    model.compile("adam", "mse")
    weights_file = args.model.replace('json', 'h5')
    model.load_weights(weights_file)

    for layer in model.layers:
        print(layer.name, layer.input, layer.output)
        print(layer.input_shape, layer.output_shape)
        print()

    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
