import keras
from keras.layers import Input, Conv2D, Add, Flatten, Dense, Lambda
import tensorflow as tf

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Split the input into three groups along the channel dimension
    def split_input(x):
        return tf.split(x, num_or_size_splits=3, axis=3)

    splitted_layers = Lambda(split_input)(input_layer)

    # Function to apply a series of convolutions
    def conv_series(x):
        conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(x)
        conv2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
        conv3 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv2)
        return conv3
    
    # Apply the convolution series to each split
    group1 = conv_series(splitted_layers[0])
    group2 = conv_series(splitted_layers[1])
    group3 = conv_series(splitted_layers[2])

    # Combine the outputs of the three groups using addition
    combined_main_path = Add()([group1, group2, group3])

    # Fuse with the original input layer through addition
    fused_output = Add()([combined_main_path, input_layer])

    # Flatten and add a fully connected layer for classification
    flatten_layer = Flatten()(fused_output)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model