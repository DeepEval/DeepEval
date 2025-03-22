import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Conv2D, DepthwiseConv2D, Conv2D, MaxPooling2D, Dropout, Flatten, Dense

def dl_model():
    
    # Input layer
    input_layer = Input(shape=(28, 28, 1))

    # Depthwise separable convolutional layer
    conv = DepthwiseConv2D(kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv)

    # Dropout layer
    conv = Dropout(rate=0.2)(conv)

    # Max pooling layer
    max_pooling = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv)

    # Dropout layer
    max_pooling = Dropout(rate=0.2)(max_pooling)

    # Flatten layer
    flatten_layer = Flatten()(max_pooling)

    # Fully connected layer
    dense1 = Dense(units=64, activation='relu')(flatten_layer)
    dense2 = Dense(units=32, activation='relu')(dense1)
    dense3 = Dense(units=10, activation='softmax')(dense2)

    # Model output
    output_layer = Dense(units=10, activation='softmax')(dense3)

    # Create the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model