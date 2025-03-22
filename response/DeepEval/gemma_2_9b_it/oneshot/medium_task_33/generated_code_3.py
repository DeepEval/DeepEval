import keras
from keras.layers import Input, Lambda, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense
import tensorflow as tf

def dl_model():     

    input_layer = Input(shape=(32, 32, 3))

    def split_channels(tensor):
      channels = tf.split(tensor, num_or_size_splits=3, axis=2)
      return channels

    split_tensor = Lambda(split_channels)(input_layer)

    # Group 1: 1x1 convolutions
    conv1_1 = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(split_tensor[0])

    # Group 2: 3x3 convolutions
    conv2_1 = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(split_tensor[1])

    # Group 3: 5x5 convolutions
    conv3_1 = Conv2D(filters=16, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(split_tensor[2])

    # Concatenate outputs
    merged_tensor = Concatenate()( [conv1_1, conv2_1, conv3_1] )

    # Flatten and fully connected layers
    flatten_layer = Flatten()(merged_tensor)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model