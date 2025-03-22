import keras
import tensorflow as tf
from keras.layers import Input, Conv2D, MaxPooling2D, Lambda, Flatten, Concatenate, DepthwiseConv2D, Dense, Reshape

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    def channel_group_extraction(input_tensor):
        channel_groups = Lambda(lambda x: tf.split(value=x, num_or_size_splits=3, axis=-1))(input_tensor)
        conv1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(channel_groups[0])
        conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(channel_groups[1])
        conv3 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(channel_groups[2])
        output_tensor = Concatenate()([conv1, conv2, conv3])
        return output_tensor

    def fc_layers(input_tensor):
        dense1 = Dense(units=64, activation='relu')(input_tensor)
        dense2 = Dense(units=64, activation='relu')(dense1)
        output_layer = Dense(units=10, activation='softmax')(dense2)
        return output_layer

    channel_group_output = channel_group_extraction(input_tensor=input_layer)
    fc_layer_output = fc_layers(input_tensor=channel_group_output)
    model = keras.Model(inputs=input_layer, outputs=fc_layer_output)

    return model