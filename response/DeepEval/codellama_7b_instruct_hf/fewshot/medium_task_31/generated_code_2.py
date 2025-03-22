import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
from keras.models import Model
import tensorflow as tf

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    def split_layer(input_tensor):
        splits = tf.split(value=input_tensor, num_or_size_splits=3, axis=-1)
        return splits

    conv1 = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
    conv3 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(conv2)
    fused = tf.concat([conv1, conv2, conv3], axis=-1)
    flattened = Flatten()(fused)
    dense1 = Dense(units=64, activation='relu')(flattened)
    dense2 = Dense(units=10, activation='softmax')(dense1)
    model = Model(inputs=input_layer, outputs=dense2)
    return model