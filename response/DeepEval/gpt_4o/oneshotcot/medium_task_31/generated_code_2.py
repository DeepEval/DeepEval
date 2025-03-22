import keras
from keras.layers import Input, Conv2D, Lambda, Concatenate, Flatten, Dense
import tensorflow as tf

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # Split the input into three groups along the channel dimension
    def split_channels(x):
        return tf.split(x, num_or_size_splits=3, axis=-1)
    
    split_layer = Lambda(split_channels)(input_layer)
    
    # Apply different convolutional kernels to each split
    conv1x1 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(split_layer[0])
    conv3x3 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(split_layer[1])
    conv5x5 = Conv2D(filters=32, kernel_size=(5, 5), padding='same', activation='relu')(split_layer[2])
    
    # Concatenate the outputs from each path
    concatenated = Concatenate()([conv1x1, conv3x3, conv5x5])
    
    # Flatten the result
    flatten_layer = Flatten()(concatenated)
    
    # Fully connected layers for classification
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)
    
    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model