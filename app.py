from flask import Flask
from flask import render_template, request
import requests
import json
import base64
import random
import os


import util

app = Flask(__name__)

BASE_DIR = os.path.dirname(__file__)

RAND_TRAIN_IMG_PATH =  os.path.join(BASE_DIR, 'images/ILSVRC/Data/DET/test')


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

    import mob_net_cls #for debugging on multiple threads

    data = request.form
    #s = time.time()
    imageJSON = json.loads(data['json'])
    img_b64 = imageJSON['img']
    #print(img_b64)
    image_np = util.decode_b64_image_to_nparr_RGB(img_b64)
    print(image_np.shape)
    res_data =  mob_net_cls.predict(image_np)
    print(res_data)
    return json.dumps( {'data': res_data}, indent=4, separators=(',', ': '))

@app.route('/get_random_image_from_cache', methods = ['GET'])
def random_image():
    rand_file = random.choice(os.listdir( RAND_TRAIN_IMG_PATH))
    rand_file_path = os.path.join(RAND_TRAIN_IMG_PATH, rand_file)
    print(rand_file_path)
    with open(rand_file_path, 'rb') as f:
        print(f)
        return base64.b64encode(f.read())


if __name__ == "__main__":
    app.run(debug=True)