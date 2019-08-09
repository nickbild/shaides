import torch
from torchvision.transforms import transforms
import numpy as np
from torch.autograd import Variable
from torchvision.models import squeezenet1_1
from io import open
import os
import sys
from PIL import Image
from train import Net
import cv2
from time import sleep
import requests
import json
import ConfigParser


config = ConfigParser.ConfigParser()
config.read(os.path.abspath(os.path.dirname(sys.argv[0])) + '/params.cfg')

hue_user = config.get('general', 'hue_user')
hue_bridge = config.get('general', 'hue_bridge')
light_id = config.get('general', 'light_id')
music_host = config.get('general', 'music_host')

img_width = 300
img_height = 300

trained_model_objects = "objects_223_9817-3576.model"
trained_model_gestures = "gestures_173_12311-6836.model"
num_classes_objects = 3
num_classes_gestures = 2

objs = []
objs.append("Google")
objs.append("Lamp")
objs.append("Nothing")

gestures = []
gestures.append("Wave")
gestures.append("Nothing")


# Load the saved models.
checkpoint_objects = torch.load(trained_model_objects)
model_objects = Net(num_classes=num_classes_objects)
model_objects.load_state_dict(checkpoint_objects)
model_objects.eval()

checkpoint_gestures = torch.load(trained_model_gestures)
model_gestures = Net(num_classes=num_classes_gestures)
model_gestures.load_state_dict(checkpoint_gestures)
model_gestures.eval()

transformation = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


def predict_image_class(image, classifier_type):
    # Preprocess the image.
    image_tensor = transformation(image).float()

    # Add an extra batch dimension since pytorch treats all images as batches.
    image_tensor = image_tensor.unsqueeze_(0)
    image_tensor.cuda()

    # Turn the input into a Variable.
    input = Variable(image_tensor)

    # Predict the class of the image.
    if classifier_type == "objects":
        output = model_objects(input)
    elif classifier_type == "gestures":
        output = model_gestures(input)

    index = output.data.numpy().argmax()
    score = output[0, index].item()

    return index, score


def gstreamer_pipeline (capture_width=3280, capture_height=2464, display_width=img_width, display_height=img_height, framerate=21, flip_method=0) :
    return ('nvarguscamerasrc ! '
    'video/x-raw(memory:NVMM), '
    'width=(int)%d, height=(int)%d, '
    'format=(string)NV12, framerate=(fraction)%d/1 ! '
    'nvvidconv flip-method=%d ! '
    'video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! '
    'videoconvert ! '
    'video/x-raw, format=(string)BGR ! appsink'  % (capture_width,capture_height,framerate,flip_method,display_width,display_height))


def toggle_lamp():
    print("TOGGLE LAMP")

    # Check to see if light is already on.
    light_state = "off"
    req = requests.get('http://' + hue_bridge + '/api/' + hue_user + '/lights')
	json_data = req.json()
    if json_data[light_id]["state"]["on"]:
        light_state = "on"

    # Prepare and send request to toggle light.
    if light_state == "off":
        json_data = json.dumps({'on': True})
    else:
        json_data = json.dumps({'on': False})

    req = requests.put('http://' + hue_bridge + '/api/' + hue_user + '/lights/' + light_id + '/state', data=json_data)
    print(req.text)


def toggle_music():
    print("TOGGLE MUSIC")
    requests.get("http://{}/toggle_play".format(music_host))


def reset_lookback():
    return [-1, -1, -1, -1, -1]


def push_lookback(recent, index):
    recent[1:] = recent[0:4]
    recent[0] = index
    return recent


def lookback_contains(recent, value):
    for i in range(len(recent)):
        if recent[i] == value:
            return True
    return False


def main():
    cap = cv2.VideoCapture(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)

    # Remember recently detected objects and gestures.
    recent_objects = reset_lookback()
    recent_gestures = reset_lookback()

    if cap.isOpened():
        while True:
            ret_val, img_in = cap.read()
            cv2.imwrite("out.jpg", img_in)
            img = Image.open('out.jpg')

            index_objects, score_objects = predict_image_class(img, "objects")
            index_gestures, score_gestures = predict_image_class(img, "gestures")

            # print("Object Class: ", objs[index_objects])
            # print("Object Score: ", score_objects)
            # print("Gestures Class: ", gestures[index_gestures])
            # print("Gestures Score: ", score_gestures)

            recent_objects = push_lookback(recent_objects, index_objects)
            recent_gestures = push_lookback(recent_gestures, index_gestures)

            # Lamp, Wave
            if lookback_contains(recent_objects, 1) and lookback_contains(recent_gestures, 0):
                recent_objects = reset_lookback()
                recent_gestures = reset_lookback()
                toggle_lamp()

            # Google, Wave
            elif lookback_contains(recent_objects, 0) and lookback_contains(recent_gestures, 0):
                recent_objects = reset_lookback()
                recent_gestures = reset_lookback()
                toggle_music()

        cap.release()
    else:
        print('Unable to open camera.')


if __name__ == "__main__":
    main()
