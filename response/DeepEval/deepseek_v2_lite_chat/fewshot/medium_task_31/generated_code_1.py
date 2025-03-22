import keras
import tensorflow as tf
from keras.layers import Input, Lambda, Conv2D, Flatten, Concatenate, Dense, Reshape

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    def block_1(input_tensor):
        split1 = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=1))(input_tensor)
        conv1_1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(split1[0])
        conv1_2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(split1[1])
        conv1_3 = Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(split1[2])
        fused_features = Concatenate()([conv1_1, conv1_2, conv1_3])
        return fused_features

    def block_2(input_tensor):
        dense1 = Dense(units=64, activation='relu')(input_tensor)
        dense2 = Dense(units=128, activation='relu')(dense1)
        reshaped = Reshape(target_shape=(4, 4, 8))(dense2)
        return reshaped

    block1_output = block_1(input_tensor=input_layer)
    block2_output = block_2(input_tensor=block1_output)

    flattened = Flatten()(block2_output)
    output_layer = Dense(units=10, activation='softmax')(flattened)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model