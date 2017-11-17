import os
import json
import random
import time
import numpy as np
import h5py
#import mob_net_cls
import tensorflow as tf


import util

random.seed(42)

s = time.time()
mob_net_bottleneck = util.mobilenet.MobileNet(weights='imagenet', include_top=False, input_shape=(224, 224, 3), pooling='avg') #consider not doing pooling and working with raw conv layer of 7x7x1024
print('time to load model', time.time() - s)

ImageDataGenerator = tf.keras.preprocessing.image.ImageDataGenerator
IMAGE_SUFFIXES = ['jpeg', 'jpg', 'png']
CONFIG_FILE_NAME = '.trainconfig'

# from keras docs
DataGenerator = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

class Trainer:

    def __init__(self, path):
        self.path = path
        self.model_path = os.path.join(self.path, 'model')
        if not os.path.isdir(self.model_path):
            os.makedirs(self.model_path)

        self.config_file_path = os.path.join(self.model_path, CONFIG_FILE_NAME)

        if os.path.isfile(self.config_file_path):
            print('file already exists')
            try:
                self.train_config = json.load(open(self.config_file_path, 'r'))
            except:
                print('JSON Decode Repulling data')
                self.train_config = self._make_train_validation()
        else:
            self.train_config = self._make_train_validation()


    def _make_train_validation(self):

        #get all uploaded files
        files = [os.path.join(self.path, f) for f in os.listdir(self.path) if f.lower().endswith('jpeg') or f.lower().endswith('jpg')]

        #randomly shuffle files
        random.shuffle(files)

        #split files at 70/30 training to validation
        N = len(files)
        split = int(0.7*N)
        train = files[:split]
        valid = files[split:]

        train_marker = {
            'timestamp': time.time(),
            'train_files' : train,
            'valid_files' : valid
        }

        json.dump(train_marker, open(self.config_file_path, 'w'), indent=4)
        print('create json file to folder', self.config_file_path)
        return train_marker

    def build_data_vectors(self):
        start = time.time()
        img_tensor = []
        for path in self.train_config['train_files']: #for large training data will need to batch process
            img_arr = util.read_image_as_nparr_RGB(path, util.INPUT_SHAPE)
            img_tensor.append(img_arr)

        img_tensor = np.array(img_tensor)
        batch_size = min(img_tensor.shape[0], 32)
        num_images = 0
        print('Batch Size: ', batch_size)
        features_tensor = None #shape: (None, 1024)

        for idx, proc_img in enumerate(DataGenerator.flow(img_tensor, batch_size=batch_size)):

            img_tensor_pp = util.mobilenet.preprocess_input(proc_img.copy().astype('float32'))
            print(img_tensor_pp.shape, img_tensor_pp.mean(), img_tensor_pp.sum()) #confirm data is changing
            features = mob_net_bottleneck.predict(img_tensor_pp, batch_size=batch_size, verbose=1)
            print(features.shape)
            num_images += features.shape[0]

            if features_tensor is None:
                features_tensor =  features
            else:
                features_tensor = np.vstack((features_tensor, features))



            if num_images > 1000:
                break
        print('Elapsed time:', time.time() - start)


        with h5py.File(os.path.join(self.model_path, 'train_data.hdf5'), 'w') as f:
            f.create_dataset('train', data=features_tensor)
        return features_tensor

    def get_fake_features(self, batch_size = 32, num_samples = 32):
        #really should split the directory into training and validation so there's no cross over.

        #select a thousand random images to use as anti-noise
        base_directory  = './images/ILSVRC/Data/DET/test'
        files_in_directory =  os.listdir('./images/ILSVRC/Data/DET/test')
        random.shuffle(files_in_directory)

        training_files = files_in_directory[:num_samples]

        i = 0
        features_tensor = None
        while i < len(training_files):
            batch_paths = training_files[i:i+batch_size]

            #work on a subset of the images so you dont overfill memory
            print(batch_paths[0], batch_paths[-1])
            batch_imgs = np.array([util.read_image_as_nparr_RGB(os.path.join(base_directory, path), util.INPUT_SHAPE) for path in batch_paths])
            batch_imgs = util.mobilenet.preprocess_input(batch_imgs.copy().astype('float32'))
            features = mob_net_bottleneck.predict(batch_imgs, batch_size=batch_size, verbose=1)
            print(features.shape)
            if features_tensor is None:
                features_tensor =  features
            else:
                features_tensor = np.vstack((features_tensor, features))
            print(features_tensor.shape)
            i += batch_size

        with h5py.File(os.path.join(self.model_path, 'train_data.hdf5'), 'a') as f:
            f.create_dataset('noise', data=features_tensor)
        return features_tensor


if __name__ == "__main__":

    tr = Trainer('./images/penny_pics')
    f_tensor = tr.build_data_vectors()
    print('f tensor shape', f_tensor.shape)
    anti_sample = tr.get_fake_features(batch_size=32, num_samples=1000)
    print('f tensor shape', anti_sample.shape)

    with h5py.File('./images/penny_pics/model/train_data.hdf5', 'r') as f:
        print(list(f.keys()))