import keras
from keras.layers import Input, AveragePooling2D, Flatten, Dense, Lambda, Concatenate, DepthwiseConv2D
from keras.models import Model

def dl_model():
    # Define the input layer
    input_layer = Input(shape=(28, 28, 1))

    # Define the first block
    block1 = AveragePooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(input_layer)
    block1 = Flatten()(block1)
    block1 = Dense(units=128, activation='relu')(block1)

    # Define the second block
    block2 = Lambda(lambda x: tf.split(x, num_or_size_splits=4, axis=-1))(block1)
    block2 = Concatenate()(block2)
    block2 = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same')(block2)
    block2 = DepthwiseConv2D(kernel_size=(5, 5), strides=(1, 1), padding='same')(block2)
    block2 = DepthwiseConv2D(kernel_size=(7, 7), strides=(1, 1), padding='same')(block2)
    block2 = Concatenate()(block2)
    block2 = Flatten()(block2)
    block2 = Dense(units=128, activation='relu')(block2)
    block2 = Dense(units=64, activation='relu')(block2)
    block2 = Dense(units=10, activation='softmax')(block2)

    # Define the model
    model = Model(inputs=input_layer, outputs=block2)

    return model