import tensorflow as tf
import numpy as np
import cv2
import time
import util
import pickle
import os
import config
#Done so that only one copy of the variable is in memory

MOB_NET = util.mobilenet.MobileNet(weights='imagenet')
MOB_NET_BOTTLENECK = util.mobilenet.MobileNet(weights='imagenet', include_top=False, input_shape=(224, 224, 3), pooling='avg')
#SVC_CLS = pickle.load(open('./is-penny-model-v1/model/sklearn-svc-acc-0.98824-2017-11-19-21-07-21.pkl', 'rb'))

def mobile_net_predict(processed_img):
    global MOB_NET
    if MOB_NET is None:
        MOB_NET = util.mobilenet.MobileNet(weights='imagenet')
    return MOB_NET.predict(processed_img)

def mobile_net_neck_predict(processed_img):
    global MOB_NET_BOTTLENECK
    if MOB_NET_BOTTLENECK is None:
        s = time.time()
        MOB_NET_BOTTLENECK = util.mobilenet.MobileNet(weights='imagenet', include_top=False, input_shape=(224, 224, 3), pooling='avg')
        print('time to load model', time.time() - s)

    return MOB_NET_BOTTLENECK.predict(processed_img)

def util_process_image(rgb_img_arr):
    img_arr = np.array(rgb_img_arr, dtype='float') #load image as np float arr
    img_arr = cv2.resize(img_arr, util.INPUT_SHAPE) #resize to input for model
    img_arr = util.mobilenet.preprocess_input(np.expand_dims(img_arr.copy(), axis=0)) #run imagenet preprocessing
    return img_arr

def run_classifier(procesed_image, top = 25):
    preds = mobile_net_predict(procesed_image)
    return  util.mobilenet.decode_predictions(preds, top=top)

_make_preds_mobile_safe = lambda preds : [{'label': label, 'prob': float(prob)}  for key, label, prob in preds[0]]

def predict(rgb_img_arr):
    img = util_process_image(rgb_img_arr)
    preds = run_classifier(img)
    return _make_preds_mobile_safe(preds)


class CustomClassifier(object):

    def __init__(self, project_name = None, model_name = None, preprocess_funcs = []):
        """

        :param project_name:
        :param model_name:
        :param preprocess_funcs: list of functions that will be applied to the input in order
        :return:
        """
        self.project_name = project_name
        self.model_name = model_name
        self.preprocess_funcs = preprocess_funcs

        self._model_path = os.path.join(config.BASE_DIR, self.project_name, 'model', self.model_name)

        self.model = None

        if self.model_name.startswith('sklearn'):
            with open(self._model_path, 'rb') as f:
                self.model = pickle.load(f)
        elif self.model_name.startswith('model'):
            #this is a keras model directory
            pass

    def _run_preprocess(self, img_arr):

        for func in self.preprocess_funcs:
            img_arr = func(img_arr)

        return img_arr

    def predict(self, img_arr):
        img_arr = self._run_preprocess(img_arr)
        return self.model.predict_proba(img_arr)

    def predict_as_dict(self, img_arr):
        pred = self.predict(img_arr)
        pred = pred.tolist()
        return dict([(idx, val) for idx, val in enumerate(pred[0])])

if __name__ == "__main__":
    cls = CustomClassifier(project_name='is-penny-model-v1',model_name='sklearn-svc-acc-0.98824-2017-11-20-21-11-24.pkl', preprocess_funcs=[util_process_image, mobile_net_neck_predict])

    img2 = util.read_image_as_nparr_RGB('./images/elephant.jpeg', shape=(224, 224))

    print(cls.predict(img2))