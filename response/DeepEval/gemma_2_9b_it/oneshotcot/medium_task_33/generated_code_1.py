import keras
from keras.layers import Input, Conv2D, Lambda, Concatenate, BatchNormalization, Flatten, Dense
import tensorflow as tf

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # Split the input channels
    def split_channels(x):
        split_tensor = tf.split(x, num_or_size_splits=3, axis=2)
        return split_tensor
    
    split_tensor = Lambda(split_channels)(input_layer)

    # Feature extraction for each channel group
    group1 = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(split_tensor[0])
    group2 = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(split_tensor[1])
    group3 = Conv2D(filters=16, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(split_tensor[2])

    # Concatenate outputs
    combined = Concatenate(axis=2)([group1, group2, group3])

    # Flatten and dense layers
    flatten_layer = Flatten()(combined)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model