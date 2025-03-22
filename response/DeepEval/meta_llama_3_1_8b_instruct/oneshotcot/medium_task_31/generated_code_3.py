import keras
from keras.layers import Input, Conv2D, Concatenate, BatchNormalization, Flatten, Dense, Lambda
from keras import backend as K
import tensorflow as tf

def dl_model():     

    input_layer = Input(shape=(32, 32, 3))
    
    # Split the input into three groups along the channel dimension
    def split_input(input_tensor):
        return Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_tensor)
    
    # Apply different convolutional kernels to each group
    split_output = split_input(input_layer)
    path1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(split_output[0])
    path2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(split_output[1])
    path3 = Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(split_output[2])
    
    # Concatenate the outputs from the three groups
    output_tensor = Concatenate()([path1, path2, path3])
    
    # Apply batch normalization and flatten the result
    bath_norm = BatchNormalization()(output_tensor)
    flatten_layer = Flatten()(bath_norm)
    
    # Pass through two fully connected layers for classification
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model