import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    AvgPool2D,
    Flatten,
    Dense
)

def dl_model():

    input_layer = Input(shape=(32, 32, 3))

    path1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    path1 = AvgPool2D()(path1)
    path1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(path1)
    path1 = AvgPool2D()(path1)

    path2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)

    path1_output = Flatten()(path1)
    path2_output = Flatten()(path2)

    merged_output = tf.keras.layers.add([path1_output, path2_output])

    output_layer = Dense(units=10, activation='softmax')(merged_output)

    model = Model(inputs=input_layer, outputs=output_layer)

    return model