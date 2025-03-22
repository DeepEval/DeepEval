import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Dropout
from keras.layers import Lambda, concatenate, Add
import tensorflow as tf

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # Block 1: Main path and Branch path
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    dropout1 = Dropout(0.2)(conv1)
    conv2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(dropout1)
    branch_path = Add()([input_layer, conv2])
    
    # Block 2: Three separable convolutional layers with different kernel sizes
    def separable_convolution(x, kernel_size):
        x = Conv2D(kernel_size=kernel_size, filters=64, padding='same', activation='relu')(x)
        x = Dropout(0.2)(x)
        return x
    
    x1, x2, x3 = tf.split(branch_path, num_or_size_splits=3, axis=-1)
    x1 = separable_convolution(x1, 1)
    x2 = separable_convolution(x2, 3)
    x3 = separable_convolution(x3, 5)
    
    block_output = concatenate([x1, x2, x3])
    
    bath_norm = BatchNormalization()(block_output)
    flatten_layer = Flatten()(bath_norm)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dropout2 = Dropout(0.2)(dense1)
    dense2 = Dense(units=64, activation='relu')(dropout2)
    dropout3 = Dropout(0.2)(dense2)
    output_layer = Dense(units=10, activation='softmax')(dropout3)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model