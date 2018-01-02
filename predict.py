import time
start = time.time()

import os
import numpy as np
import pandas as pd
from glob import glob
import cv2
from utils import path_to_tensor
from keras.applications import *
from model import get_features, linear, pretrain
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True



inception_sz = (1536,)
X = np.zeros((299, 299, 3), dtype=np.float32)
sorted_breed = np.load('breed_sorted.npy')

pretrain_model = pretrain(InceptionResNetV2)
classifier = linear(inp_sz=inception_sz, out_sz=120)
classifier.load_weights('weights.best.inceptresnetv2.comb.hdf5')

pred_images = np.array(glob("pic_pred/*"))

for dog_pic in pred_images:
    time_loop = time.time()
    tensor = path_to_tensor(dog_pic)
    time_vectorize = time.time()
    features = pretrain_model.predict(tensor, batch_size=256, verbose=1)
    time_feature = time.time()
    y_pred = classifier.predict(features, batch_size=1)
    pred = np.argmax(y_pred, axis=1)
    time_classify = time.time()
    print(dog_pic, "is a ", sorted_breed[pred])
    print("vectorize time", time_vectorize - time_loop)
    print("forward pass time", time_feature - time_vectorize)
    print("classify time", time_classify - time_feature)
    print("total time per img", time_classify - time_loop)



end = time.time()
print (end - start)