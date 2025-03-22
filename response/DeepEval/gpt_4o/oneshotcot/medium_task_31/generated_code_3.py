import keras
from keras.layers import Input, Lambda, Conv2D, Concatenate, Flatten, Dense
import tensorflow as tf

def dl_model():
    
    # Step 1: Add input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Step 2: Split input into three groups along the channel dimension
    def split_channels(x):
        return tf.split(x, num_or_size_splits=3, axis=-1)
    
    split_layer = Lambda(split_channels)(input_layer)
    
    # Step 3: Apply convolutional layers with different kernels
    conv1x1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(split_layer[0])
    conv3x3 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(split_layer[1])
    conv5x5 = Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(split_layer[2])
    
    # Step 4: Concatenate the outputs from these three groups
    concatenated = Concatenate()([conv1x1, conv3x3, conv5x5])
    
    # Step 5: Flatten the fused features
    flatten_layer = Flatten()(concatenated)
    
    # Step 6: Add dense layers for classification
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)
    
    # Step 7: Build and return the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model