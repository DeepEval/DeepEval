import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Lambda, Reshape, Permute, DepthwiseConv2D

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Main Path
    def block(input_tensor):
        splits = tf.split(input_tensor, 3, axis=3)
        conv1s = [Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(x) for x in splits]
        concat = Concatenate(axis=3)(conv1s)
        return concat

    block1 = block(input_layer)

    block2 = Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(block1)
    block2 = Reshape((32, 32, 1, 64))(block2)
    block2 = Permute((0, 3, 1, 2))(block2)
    block2 = Reshape((32, 32, 64))(block2)

    block3 = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(block2)

    # Branch Path
    branch = AveragePooling2D(pool_size=(32, 32), strides=(32, 32), padding='valid')(input_layer)
    branch = Flatten()(branch)
    branch = Dense(units=10, activation='softmax')(branch)

    # Concatenation and Output Layer
    concat = Concatenate()([block3, branch])
    output_layer = Dense(units=10, activation='softmax')(concat)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model