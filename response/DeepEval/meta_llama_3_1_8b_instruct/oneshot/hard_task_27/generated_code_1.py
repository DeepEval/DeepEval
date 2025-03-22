import keras
from keras.layers import Input, DepthwiseConv2D, Conv2D, BatchNormalization, Add, GlobalAveragePooling2D, Dense, LayerNormalization
from keras.layers import Concatenate, Flatten
from keras import regularizers
import tensorflow as tf

def dl_model():     

    input_layer = Input(shape=(32, 32, 3))
    dw_conv = DepthwiseConv2D(kernel_size=(7, 7), strides=(1, 1), padding='same')(input_layer)
    dw_conv = LayerNormalization()(dw_conv)
    dw_conv = tf.nn.relu(dw_conv)

    dw_conv = Conv2D(filters=3, kernel_size=(1, 1), strides=(1, 1), padding='same')(dw_conv)

    def block(input_tensor):

        conv1 = Dense(units=3, activation='relu')(input_tensor)
        output_tensor = Concatenate()([conv1, input_tensor])

        return output_tensor
        
    block_output = block(dw_conv)
    block_output = GlobalAveragePooling2D()(block_output)
    dense1 = Dense(units=512, activation='relu')(block_output)
    dense2 = Dense(units=512, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    # Combine the original input with the processed features through an addition operation
    original_input = Input(shape=(32, 32, 3))
    combined = Add()([original_input, dw_conv])
    combined = GlobalAveragePooling2D()(combined)
    combined = Dense(units=512, activation='relu')(combined)
    combined = Dense(units=10, activation='softmax')(combined)

    # Create a new model that takes the original input and outputs the classification result
    model = keras.Model(inputs=original_input, outputs=combined)

    return model