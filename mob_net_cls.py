import tensorflow as tf
import numpy as np
import cv2
import time

s = time.time()
mobilenet = tf.keras.applications.mobilenet
mob_net = mobilenet.MobileNet(weights='imagenet')
print('time to load model', time.time() - s)

INPUT_SHAPE = (224, 224)

def process_image(rgb_img_arr):
    img_arr = np.array(rgb_img_arr, dtype='float') #load image as np float arr
    img_arr = cv2.resize(img_arr, INPUT_SHAPE) #resize to input for model
    img_arr = mobilenet.preprocess_input(np.expand_dims(img_arr.copy(), axis=0)) #run imagenet preprocessing
    return img_arr

def run_classifier(procesed_image, top = 25):
    preds = mob_net.predict(procesed_image)
    return  mobilenet.decode_predictions(preds, top=top)

_make_preds_mobile_safe = lambda preds : [{'label': label, 'prob': float(prob)}  for key, label, prob in preds[0]]

def predict(rgb_img_arr):
    img = process_image(rgb_img_arr)
    preds = run_classifier(img)
    return _make_preds_mobile_safe(preds)

