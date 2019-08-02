import cv2
from time import sleep
import sys


img_width = 300
img_height = 300
batch = "1"
dir = "data/train/lamp1"


def gstreamer_pipeline (capture_width=3280, capture_height=2464, display_width=img_width, display_height=img_height, framerate=21, flip_method=0) :
    return ('nvarguscamerasrc ! '
    'video/x-raw(memory:NVMM), '
    'width=(int)%d, height=(int)%d, '
    'format=(string)NV12, framerate=(fraction)%d/1 ! '
    'nvvidconv flip-method=%d ! '
    'video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! '
    'videoconvert ! '
    'video/x-raw, format=(string)BGR ! appsink'  % (capture_width,capture_height,framerate,flip_method,display_width,display_height))


# Capture [num_images] images, spaced [delay_sec] seconds apart.
def save_frame_sequence(num_images, delay_sec):
    if cap.isOpened():
        for i in range(num_images+20):
            ret_val, img = cap.read()
            if i >= 20: # Skip first 20 to let camera get lighting straigtened out.
                cv2.imwrite(dir + "/img_" + str(i) + "_" + batch + ".jpg", img)
            sleep(delay_sec)

        cap.release()
    else:
        print('Unable to open camera.')


if __name__ == '__main__':
    cap = cv2.VideoCapture(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)
    sleep(5)
    print("Go!")
    save_frame_sequence(10, 0.2)
