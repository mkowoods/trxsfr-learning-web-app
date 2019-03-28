# SPECS:
 - Python 3.5
 - Tesnorflow 1.4


# FUTURE OPTIMIZATIONS 
 - quantize the model:  https://www.tensorflow.org/performance/quantization
 - can you serve the model faster using different compilation techniques?
 
 
 Need some random sample data to train the model as well
 
# DATASET:
https://www.kaggle.com/c/imagenet-object-detection-challenge/data

# Load Balancing: 
https://cloud.google.com/compute/docs/load-balancing/network/example

## Memory footprint
tests are based on importing tensorflow and loading the  weights for imagenet
```
import tensorflow as tf
mobilenet = tf.keras.applications.mobilenet
```

- Loading mobilenet with alpha @ 0.25
    - `wts = mobilenet.MobileNet(weights='imagenet', alpha = 0.25)`
    - Memory 138.4MB

- Loading mobilenet with alpha @ 1.0
    - `wts = mobilenet.MobileNet(weights='imagenet', alpha = 1.0)`
    - Memory 157.7MB
- Loading mobilenet with alpah @ 0.25, no top and image size = 128 by 128
    - `wts = mobilenet.MobileNet(weights='imagenet', include_top = False, alpha = 0.25, input_shape = (128, 128, 3))'
    - Memory: 137.6MB
    
    
`nohup ~/miniconda/envs/trxsfr-learn-web/bin/python ~/trxsfr-learning-web-app/app.py &`
