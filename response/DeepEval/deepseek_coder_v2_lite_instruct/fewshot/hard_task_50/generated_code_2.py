import keras
from keras.layers import Input, MaxPooling2D, Lambda, Flatten, Concatenate, DepthwiseConv2D, Dense, Reshape, Dropout
import tensorflow as tf

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 with 3 color channels

    def block_1(input_tensor):
        maxpool1 = MaxPooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(input_tensor)
        flatten1 = Flatten()(maxpool1)
        maxpool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(input_tensor)
        flatten2 = Flatten()(maxpool2)
        maxpool3 = MaxPooling2D(pool_size=(4, 4), strides=(4, 4), padding='same')(input_tensor)
        flatten3 = Flatten()(maxpool3)
        output_tensor = Concatenate()([flatten1, flatten2, flatten3])
        output_tensor = Dropout(0.5)(output_tensor)  # Adding dropout for regularization
        return output_tensor

    def block_2(input_tensor):
        inputs_groups = Lambda(lambda x: tf.split(value=x, num_or_size_splits=4, axis=-1))(input_tensor)
        conv1 = DepthwiseConv2D(kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(inputs_groups[0])
        conv2 = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(inputs_groups[1])
        conv3 = DepthwiseConv2D(kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(inputs_groups[2])
        conv4 = DepthwiseConv2D(kernel_size=(7, 7), strides=(1, 1), padding='same', activation='relu')(inputs_groups[3])
        output_tensor = Concatenate()([conv1, conv2, conv3, conv4])
        return output_tensor

    block1_output = block_1(input_tensor=input_layer)
    dense = Dense(units=128, activation='relu')(block1_output)
    reshaped = Reshape(target_shape=(4, 4, 8))(dense)  # Adjusted based on the number of groups and features
    block2_output = block_2(input_tensor=reshaped)

    flatten = Flatten()(block2_output)
    output_layer = Dense(units=10, activation='softmax')(flatten)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model