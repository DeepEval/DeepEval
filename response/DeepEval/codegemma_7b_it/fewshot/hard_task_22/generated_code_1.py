import keras
import tensorflow as tf
from keras.layers import Input, Lambda, Concatenate, Conv2D, MaxPooling2D, Flatten, Dense

def dl_model():

    input_layer = Input(shape=(32, 32, 3))

    def main_path(input_tensor):
        groups = Lambda(lambda x: tf.split(value=x, num_or_size_splits=3, axis=-1))(input_tensor)
        conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(groups[0])
        conv2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(groups[1])
        conv3 = Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(groups[2])
        concat = Concatenate()([conv1, conv2, conv3])
        return concat

    def branch_path(input_tensor):
        conv = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        return conv

    main_output = main_path(input_layer)
    branch_output = branch_path(input_layer)
    concat = Add()([main_output, branch_output])

    flatten = Flatten()(concat)
    dense1 = Dense(units=128, activation='relu')(flatten)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model