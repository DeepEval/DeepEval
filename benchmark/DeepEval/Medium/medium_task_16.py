import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Conv2D,Lambda, AveragePooling2D, Concatenate, Flatten, Dense

def dl_model():

    input_layer = Input(shape=(32, 32, 3))

    num_groups = 3
    input_groups = Lambda(function=lambda x: tf.split(value=x, num_or_size_splits=num_groups, axis=-1))(input_layer)
    input_channels = input_layer.shape[-1]

    conv1 = Conv2D(filters=input_channels // num_groups, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_groups[0])
    output_1 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv1)

    conv2 = Conv2D(filters=input_channels // num_groups, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_groups[1])
    output_2 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv2)

    conv3 = Conv2D(filters=input_channels // num_groups, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_groups[2])
    output_3 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv3)

    conca = Concatenate()([output_1, output_2, output_3])

    flatten_output = Flatten()(conca)
    dense_1_output = Dense(units=128, activation='relu')(flatten_output)
    output_layer = Dense(units=10, activation='softmax')(dense_1_output)

    model = Model(inputs=input_layer, outputs=output_layer)

    return model
