import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense
from keras.layers import concatenate  # Importing the correct concatenate function
import tensorflow as tf  # Importing TensorFlow to avoid potential issues with Keras

def dl_model():
    
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 dataset has 32x32 images with 3 color channels
    conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(input_layer)

    def pooling_layer(window_size):
        return MaxPooling2D(pool_size=(window_size, window_size), strides=(window_size, window_size), padding='valid')

    pool1 = pooling_layer(1)(conv)
    pool2 = pooling_layer(2)(pool1)
    pool3 = pooling_layer(4)(pool2)

    flat1 = Flatten()(pool1)
    flat2 = Flatten()(pool2)
    flat3 = Flatten()(pool3)

    # The correct function to use is concatenate from keras.layers, but it can also be called as the correct function from the tensorflow.keras.layers module
    from tensorflow.keras.layers import concatenate  # Importing the correct concatenate function
    combined_features = concatenate([flat1, flat2, flat3])

    dense1 = Dense(units=128, activation='relu')(combined_features)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model