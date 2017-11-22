import base64
import cv2
import os
import numpy as np
import tensorflow as tf
#from tensorflow.contrib.keras.api.keras.applications.mobilenet import MobileNet


mobilenet = tf.keras.applications.mobilenet

INPUT_SHAPE = (224, 224)


def decode_b64_image_to_nparr_RGB(img_b64):
    img_decodedb64 = base64.b64decode(img_b64)
    nparr = np.fromstring(img_decodedb64, np.uint8)
    nparr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    nparr = cv2.cvtColor(nparr, cv2.COLOR_BGR2RGB)
    return nparr


def read_image_as_nparr_RGB(path, shape = None):
    img_BGR = cv2.imread(path)
    if shape is not None:
        img_BGR = cv2.resize(img_BGR, shape)
    return cv2.cvtColor(img_BGR, cv2.COLOR_BGR2RGB)


