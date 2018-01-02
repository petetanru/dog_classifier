from keras.layers import *
from keras.models import *
from keras.applications import *
from keras.applications.inception_resnet_v2 import preprocess_input

def get_features(MODEL, data):
    h, w = data.shape[1], data.shape[2]
    cnn_model = MODEL(include_top=False, input_shape=(h, w, 3), weights='imagenet')
    inputs = Input((h, w, 3))
    x = inputs
    x = Lambda(preprocess_input, name='preprocessing')(x)
    x = cnn_model(x)
    x = GlobalAveragePooling2D()(x)
    cnn_model = Model(inputs, x)
    features = cnn_model.predict(data, batch_size=256, verbose=1)
    return features

def pretrain(MODEL):
    h, w = 299, 299
    cnn_model = MODEL(include_top=False, input_shape=(h, w, 3), weights='imagenet')
    inputs = Input((h, w, 3))
    x = inputs
    x = Lambda(preprocess_input, name='preprocessing')(x)
    x = cnn_model(x)
    x = GlobalAveragePooling2D()(x)
    cnn_model = Model(inputs, x)
    return cnn_model

def linear(inp_sz, out_sz):
    inputs = Input(inp_sz)
    x = inputs
    # x = BatchNormalization()(x)
    # x = Dense(1024)(x)
    # x = BatchNormalization()(x)
    # x = Dense(256)(x)
    # x = Activation('relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(out_sz, activation='softmax')(x)
    model = Model(inputs, x)
    return model
