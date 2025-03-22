import keras
import tensorflow as tf
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, Dense, Flatten, Lambda

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    def split_and_process(input_tensor):
        split_layers = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_tensor)
        conv1 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(split_layers[0])
        conv2 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(split_layers[1])
        conv3 = Conv2D(filters=32, kernel_size=(5, 5), padding='same', activation='relu')(split_layers[2])
        return Concatenate()([conv1, conv2, conv3])

    processed_output = split_and_process(input_tensor=input_layer)

    dense1 = Dense(units=128, activation='relu')(processed_output)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model