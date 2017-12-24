import os
import numpy as np
import pandas as pd
from glob import glob
import cv2
from utils import path_to_tensor
from keras.applications import *
from model import get_features, linear
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

X = np.zeros((299, 299, 3), dtype=np.float32)
breed = np.load('dog_names.npy')

pred_images = np.array(glob("pic_pred/*"))
for dog_pic in pred_images:
    tensor = path_to_tensor(dog_pic)
    inception_features = get_features(InceptionResNetV2, tensor)

    # TODO: state this before the loop, change model file too
    model = linear(inception_features, out_sz=120)
    # model.load_weights('weights.best.inceptresnetv2.hdf5')

    y_pred = model.predict(inception_features, batch_size=1)
    pred = np.argmax(y_pred, axis=1)
    print(dog_pic, "is a ", breed[pred])



