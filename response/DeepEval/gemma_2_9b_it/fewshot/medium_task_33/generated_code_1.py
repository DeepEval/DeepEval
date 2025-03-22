import keras
import tensorflow as tf
from keras.layers import Input, Lambda, Conv2D, Concatenate, Flatten, Dense

def dl_model():
    
    input_layer = Input(shape=(32, 32, 3))

    def split_and_process(input_tensor):
        channels = tf.split(value=input_tensor, num_or_size_splits=3, axis=-1)
        conv1x1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(channels[0])
        conv3x3 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(channels[1])
        conv5x5 = Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(channels[2])
        return Concatenate()([conv1x1, conv3x3, conv5x5])

    processed_tensor = split_and_process(input_layer)
    flatten_layer = Flatten()(processed_tensor)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model