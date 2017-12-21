from flask import Flask
from flask import render_template, request
import requests
import json
import base64
import random
import os
import config
import mob_net_cls
import time
import logging
import util

app = Flask(__name__)

BASE_DIR = config.BASE_DIR

RAND_TRAIN_IMG_PATH =  os.path.join(BASE_DIR, 'images/ILSVRC/Data/DET/test')

s = time.time()
PENNY_MODEL = mob_net_cls.CustomClassifier(project_name = 'is-penny-model-v1',
                                           model_name='sklearn-svc-acc-0.98824-2017-11-20-21-11-24.pkl',
                                           preprocess_funcs=[mob_net_cls.util_process_image, mob_net_cls.mobile_net_neck_predict])
s1 = time.time()
print('Loading PENNY_MODEL', s1 - s)


_ = PENNY_MODEL.predict(util.read_image_as_nparr_RGB('./images/elephant.jpeg', shape=(224, 224)))
print('Time to Load Model', time.time() - s1)


#print(res)
logging.basicConfig(level= logging.INFO)
logging.info('Called app.py')

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/get_remote_image')
def get_remote_image():
    image_url = request.args.get('image_url')
    print(image_url)
    resp = requests.get(image_url)
    if resp.status_code == 200:
        return base64.b64encode(resp.content)

@app.route('/predict_mobilenet', methods = ['POST'])
def get_results():
    s = time.time()
    data = request.form
    imageJSON = json.loads(data['json'])
    img_b64 = imageJSON['img']
    s1 = time.time()
    logging.info('Elapsed Time to Load: {time:.3f}'.format(time =  time.time() - s))
    image_np = util.decode_b64_image_to_nparr_RGB(img_b64)
    #print(image_np)
    res_data = mob_net_cls.predict(image_np)
    logging.info('Time To Predict: {time:.3f}'.format(time =  (time.time() - s1)))
    return json.dumps( {'data': res_data}, indent=4, separators=(',', ': '))

@app.route('/get_random_image_from_cache', methods = ['GET'])
def random_image():
    rand_file = random.choice(os.listdir( RAND_TRAIN_IMG_PATH))
    rand_file_path = os.path.join(RAND_TRAIN_IMG_PATH, rand_file)
    print(rand_file_path)
    with open(rand_file_path, 'rb') as f:
        print(f)
        return base64.b64encode(f.read())


@app.route('/api/<model_name>', methods = ['POST'])
def load_model(model_name):
    s = time.time()

    image_json = json.loads(request.form['json'])
    img_b64 = image_json['img']
    image_np = util.decode_b64_image_to_nparr_RGB(img_b64)
    s1 = time.time()
    print('Time to load image', s1 - s)

    #image_np = util.read_image_as_nparr_RGB('./images/elephant.jpeg', shape=(224, 224))
    #print(image_np.shape)
    res_data = PENNY_MODEL.predict_as_dict(image_np)
    print('Time to predict', time.time() - s1)
    print('Elapsed Time', time.time() - s)
    return json.dumps({'data': res_data}, indent=4, separators=(',', ': '))

if __name__ == "__main__":
    app.run(debug=False)