import torch
from torchvision.transforms import transforms
import numpy as np
from torch.autograd import Variable
from torchvision.models import squeezenet1_1
from io import open
import os
from PIL import Image
from train import Net
import cv2
from time import sleep
import requests


img_width = 300
img_height = 300
hue_bridge = "192.168.1.113:5000"
trained_model_objects = "objects_223_9817-3576.model"
trained_model_gestures = "gestures_422_5847-2803.model"
num_classes_objects = 3
num_classes_gestures = 2

objs = []
objs.append("Google")
objs.append("Lamp")
objs.append("Nothing")


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


if __name__ == "__main__":
    cap = cv2.VideoCapture(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)

    if cap.isOpened():
        while True:
            ret_val, img = cap.read()
            index_objects, score_objects = predict_image_class(img, "objects")
            index_gestures, score_gestures = predict_image_class(img, "gestures")

            print("Object Class: ", objs[index_objects])
            print("Object Score: ", score_objects)
            print("Gestures Class: ", index_gestures)
            print("Gestures Score: ", score_gestures)

            # if index == 0 and lastIndex == 0 and score > 10:
            #     #print("Backwards")
            #     requests.get("http://{}/backward".format(doom_host))


        cap.release()
    else:
        print('Unable to open camera.')
