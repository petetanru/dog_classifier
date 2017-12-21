import cv2
import numpy as np
import pandas as pd
from sklearn.datasets import load_files
from keras.utils import np_utils
import numpy as np
from glob import glob
from sklearn.utils import shuffle

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from keras.preprocessing import image
from tqdm import tqdm
from keras.layers import *
from keras.models import *
from keras.applications import *
from keras.optimizers import *
from keras.regularizers import *
from keras.applications.inception_resnet_v2 import preprocess_input
from keras.callbacks import ModelCheckpoint

def load_dataset(path):
    data = load_files(path)
    dog_files = np.array(data['filenames'])
    dog_targets = np_utils.to_categorical(np.array(data['target']), 120)
    return dog_files, dog_targets

train_files, train_targets = load_dataset('stanford/Images_re')

print("first file before shuffle", train_files[0], train_targets[0])
train_files, train_targets = shuffle(train_files, train_targets, random_state=0)
print("first file after shuffle", train_files[0], train_targets[0])

dog_names = [item[29:-1] for item in sorted(glob("stanford/Images_re/*/"), reverse=True)]
np.save('dog_names', dog_names)
print(dog_names)

# print statistics about the dataset
print('There are %d total dog categories.' % len(dog_names))
print('There are %d training dog images.' % len(train_files))
n_class = len(dog_names)

def path_to_tensor(img_path):
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(299, 299))
    # convert PIL.Image.Image type to 3D tensor with shape (299, 299, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)

def paths_to_tensor(img_paths):
    list_of_tensors = [path_to_tensor(img_path) for img_path in tqdm(img_paths)]
    return np.vstack(list_of_tensors)

# train_tensors = paths_to_tensor(train_files).astype('float32')/255
train_tensors = paths_to_tensor(train_files)

h, w = 299, 299
def get_features(MODEL, data):
    cnn_model = MODEL(include_top=False, input_shape=(h, w, 3), weights='imagenet')
    inputs = Input((h, w, 3))
    x = inputs
    x = Lambda(preprocess_input, name='preprocessing')(x)
    x = cnn_model(x)
    x = GlobalAveragePooling2D()(x)
    cnn_model = Model(inputs, x)
    features = cnn_model.predict(data, batch_size=256, verbose=1)
    return features

inception_features = get_features(InceptionResNetV2, train_tensors)

inputs = Input(inception_features.shape[1:])
x = inputs
x = Dropout(0.5)(x)
x = Dense(n_class, activation='softmax')(x)
model = Model(inputs, x)
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

checkpointer = ModelCheckpoint(filepath='weights.best.inceptresnetv2.hdf5',
                               verbose=1, save_best_only=True)

h = model.fit(inception_features, train_targets, shuffle=True, batch_size=256, epochs=100, validation_split=0.05, verbose=2, callbacks=[checkpointer])


