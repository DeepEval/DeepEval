import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Lambda, SeparableConv2D, Concatenate, Dense

def dl_model():

    input_layer = Input(shape=(32, 32, 3))
    inputs_groups = Lambda(function=lambda x: tf.split(value=x, num_or_size_splits=3, axis=-1))(input_layer)

    conv1 = SeparableConv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(inputs_groups[0])
    conv2 = SeparableConv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(inputs_groups[1])
    conv3 = SeparableConv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(inputs_groups[2])
    output_tensor = Concatenate()([conv1, conv2, conv3])

    dense1 = Dense(units=10, activation='softmax')(output_tensor)
    dense2 = Dense(units=10, activation='softmax')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = Model(inputs=input_layer, outputs=output_layer)

    return model
