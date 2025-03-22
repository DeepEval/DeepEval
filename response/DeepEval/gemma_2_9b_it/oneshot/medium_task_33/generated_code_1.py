import tensorflow as tf
from tensorflow import keras
from keras.layers import Input, Conv2D, Lambda, Concatenate, Flatten, Dense

def dl_model(): 
    input_layer = Input(shape=(32, 32, 3))

    def split_channels(input_tensor):
      channels = tf.split(input_tensor, num_or_size_splits=3, axis=2)
      return channels

    split_tensor = Lambda(split_channels)(input_layer)

    # Process each channel group
    group1 = Conv2D(filters=16, kernel_size=(1, 1), activation='relu')(split_tensor[0])
    group1 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(group1)

    group2 = Conv2D(filters=16, kernel_size=(1, 1), activation='relu')(split_tensor[1])
    group2 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(group2)

    group3 = Conv2D(filters=16, kernel_size=(1, 1), activation='relu')(split_tensor[2])
    group3 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(group3)

    # Concatenate processed groups
    concatenated = Concatenate(axis=2)([group1, group2, group3])

    # Flatten and dense layers
    flatten = Flatten()(concatenated)
    dense1 = Dense(units=128, activation='relu')(flatten)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)
    return model