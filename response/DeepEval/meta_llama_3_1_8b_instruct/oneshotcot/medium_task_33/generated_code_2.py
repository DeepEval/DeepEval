import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Lambda
from keras import backend as K
import tensorflow as tf

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # Assuming 32x32x3 input images
    
    # Split the input into three channel groups
    def split_input(x):
        return tf.split(x, num_or_size_splits=3, axis=3)
    split_input_layer = Lambda(split_input)(input_layer)

    # Define the separable convolutional layers
    def separable_convolution(path, filter_size):
        return Conv2D(kernel_size=(filter_size, filter_size), padding='same', activation='relu', 
                      kernel_initializer='he_normal')(path)

    path1 = separable_convolution(split_input_layer[0], 1)
    path2 = separable_convolution(split_input_layer[1], 3)
    path3 = separable_convolution(split_input_layer[2], 5)

    # Concatenate the outputs from the three groups
    output_tensor = Concatenate()([path1, path2, path3])

    # Apply batch normalization and flatten the result
    batch_norm = BatchNormalization()(output_tensor)
    flatten_layer = Flatten()(batch_norm)

    # Define the three fully connected layers
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model