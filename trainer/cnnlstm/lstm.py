from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.layers import Dense, Input
from keras.layers.pooling import GlobalAveragePooling2D
from keras.layers.recurrent import LSTM
from keras.layers.wrappers import TimeDistributed
from keras.optimizers import Nadam
import tensorflow as tf

def create_model(frames, channels, width, height, classes):
    video = Input(shape=(frames, channels, width, height))
    cnn_base = VGG16(
        input_shape=(channels, width, height), weights="imagenet", include_top=False
    )
    cnn_out = GlobalAveragePooling2D()(cnn_base.output)
    cnn = Model(inputs=cnn_base.input, outputs=cnn_out)
    cnn.trainable = False
    encoded_frames = TimeDistributed(cnn)(video)
    encoded_sequence = LSTM(256)(encoded_frames)
    hidden_layer = Dense(1024, activation="relu")(encoded_sequence)
    outputs = Dense(classes, activation="softmax")(hidden_layer)
    model = Model([video], outputs)
    optimizer = Nadam(
        lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004
    )
    model.compile(
        loss="categorical_crossentropy",
        optimizer=optimizer,
        metrics=["categorical_accuracy"],
    )
    model(tf.ones((1, frames, channels, width, height)))
    return model
