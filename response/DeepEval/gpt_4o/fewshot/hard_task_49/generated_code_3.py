import keras
import tensorflow as tf
from keras.layers import Input, AveragePooling2D, Lambda, Flatten, Concatenate, DepthwiseConv2D, Dense, Reshape

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    def block_1(input_tensor):
        # Average pooling with different scales
        pool1 = AveragePooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(input_tensor)
        flatten1 = Flatten()(pool1)
        pool2 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(input_tensor)
        flatten2 = Flatten()(pool2)
        pool3 = AveragePooling2D(pool_size=(4, 4), strides=(4, 4), padding='same')(input_tensor)
        flatten3 = Flatten()(pool3)
        # Concatenate flattened outputs
        output_tensor = Concatenate()([flatten1, flatten2, flatten3])
        return output_tensor

    def block_2(input_tensor):
        # Split input into four parts
        split_inputs = Lambda(lambda x: tf.split(x, num_or_size_splits=4, axis=-1))(input_tensor)
        # Apply depthwise separable convolutions
        conv1 = DepthwiseConv2D(kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(split_inputs[0])
        conv2 = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(split_inputs[1])
        conv3 = DepthwiseConv2D(kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(split_inputs[2])
        conv4 = DepthwiseConv2D(kernel_size=(7, 7), strides=(1, 1), padding='same', activation='relu')(split_inputs[3])
        # Concatenate outputs from different paths
        output_tensor = Concatenate()([conv1, conv2, conv3, conv4])
        return output_tensor

    # Build model
    block1_output = block_1(input_layer)
    dense = Dense(units=64, activation='relu')(block1_output)
    reshaped = Reshape(target_shape=(4, 4, 4))(dense)
    block2_output = block_2(reshaped)
    
    flatten = Flatten()(block2_output)
    output_layer = Dense(units=10, activation='softmax')(flatten)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model