# SPECS:
 - Python 3.5
 - Tesnorflow 1.4


 # TODO
 https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html
https://hackernoon.com/creating-insanely-fast-image-classifiers-with-mobilenet-in-tensorflow-f030ce0a2991
 need to add a batch learner for classifying the case
 
 compile tensorflow from scratch https://stackoverflow.com/questions/41293077/how-to-compile-tensorflow-with-sse4-2-and-avx-instructions
 https://gist.github.com/bage79/ef955a460e33555830ea9217c5e2e925
 
# FUTURE OPTIMIZATIONS 
 - quantize the model:  https://www.tensorflow.org/performance/quantization
 - can you serve the model faster using different compilation techniques?
 

## sample remote images 
 https://www.cesarsway.com/sites/newcesarsway/files/styles/large_article_preview/public/Common-dog-behaviors-explained.jpg?itok=FSzwbBoi
 https://fm.cnbc.com/applications/cnbc.com/resources/img/editorial/2017/05/12/104466932-PE_Color.240x240.jpg?v=1494613853
 

 
 https://gking.harvard.edu/files/0s.pdf
 https://groups.google.com/forum/#!topic/keras-users/MUO6v3kRHUw
 https://medium.com/@timanglade/how-hbos-silicon-valley-built-not-hotdog-with-mobile-tensorflow-keras-react-native-ef03260747f3
 https://www.tomstall.com/content/create-a-globally-distributed-tensorflow-serving-cluster-with-nearly-no-pain/
 https://gist.github.com/avloss/01e43d208fbdb2c5b4f9b50e71617cc8
 
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