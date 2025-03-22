import keras
import tensorflow as tf
from keras.layers import Input, Conv2D, Add, Concatenate, Dense, Flatten

def dl_model():

    input_layer = Input(shape=(32, 32, 64))

    def reduce_dim(input_tensor):
        conv1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        return Concatenate()([conv1, conv2])

    main_path_output = reduce_dim(input_tensor=input_layer)

    def expand_dims(input_tensor):
        conv3 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        return tf.expand_dims(conv3, axis=-1)

    branch_path_output = expand_dims(input_tensor=input_layer)

    combined_output = Add()([main_path_output, branch_path_output])

    def classify(input_tensor):
        flatten = Flatten()(input_tensor)
        dense1 = Dense(units=128, activation='relu')(flatten)
        dense2 = Dense(units=10, activation='softmax')(dense1)
        return dense2

    final_output = classify(input_tensor=combined_output)

    model = keras.Model(inputs=input_layer, outputs=final_output)

    return model