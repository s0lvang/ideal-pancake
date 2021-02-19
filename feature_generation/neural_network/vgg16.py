from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.models import Model, Sequential
from keras.layers import Dense, Input
from keras.layers.pooling import GlobalAveragePooling2D
from keras.layers.recurrent import LSTM
from keras.layers.wrappers import TimeDistributed
from keras.optimizers import Adam
from tensorflow.keras import backend as K
import tensorflow as tf
import numpy as np


def create_model_factory(frames, channels, width, height, classes):
    return lambda: create_transfer_learning_model(
        frames, channels, width, height, classes
    )


def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))


def create_transfer_learning_model(frames, width, height, channels, classes):
    print("frames", frames)
    print("channels", channels)
    print("width", width)
    print("height", height)
    print("classes", classes)
    video = Input(shape=(frames, width, height, channels))
    cnn_base = VGG16(
        input_shape=(width, height, channels), weights="imagenet", include_top=False
    )
    cnn_out = GlobalAveragePooling2D()(cnn_base.output)
    cnn = Model(inputs=cnn_base.input, outputs=cnn_out)
    cnn.trainable = False
    encoded_frames = TimeDistributed(cnn)(video)
    encoded_sequence = LSTM(10)(encoded_frames)
    hidden_layer = Dense(50, activation="relu")(encoded_sequence)
    outputs = Dense(1, activation="linear")(hidden_layer)
    model = Model([video], outputs)
    optimizer = Adam(lr=0.0001)
    model.compile(loss=root_mean_squared_error, optimizer=optimizer, metrics=["mse"])
    print(model.summary())
    return model


def extract_features_from_vgg16(subjects):
    cnn_base = VGG19(
        input_shape=subjects.shape[2:], weights="imagenet", include_top=False
    )
    predictions = np.array([cnn_base.predict(images) for images in subjects])
    flattened_subjects = predictions.reshape((*predictions.shape[:1], -1))
    return flattened_subjects
