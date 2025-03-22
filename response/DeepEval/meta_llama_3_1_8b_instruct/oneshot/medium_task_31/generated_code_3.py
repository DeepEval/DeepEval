import keras
from keras.layers import Input, Conv2D, Concatenate, Lambda, Flatten, Dense
import tensorflow as tf

def dl_model():     

    input_layer = Input(shape=(32, 32, 3))
    
    # Split the input along the channel dimension into three groups
    def split_input(input_tensor):
        return tf.split(input_tensor, num_or_size_splits=3, axis=-1)

    split_layer = Lambda(split_input)(input_layer)
    
    # Apply different convolutional kernels for multi-scale feature extraction
    conv1x1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(split_layer[0])
    conv3x3 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(split_layer[1])
    conv5x5 = Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(split_layer[2])
    
    # Concatenate the outputs from these three groups
    output_tensor = Concatenate()([conv1x1, conv3x3, conv5x5])
    
    # Flatten the concatenated output into a one-dimensional vector
    flatten_layer = Flatten()(output_tensor)
    
    # Pass the flattened output through two fully connected layers for classification
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model