import keras
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, AveragePooling2D, Concatenate, Flatten, Dense

def dl_model():
    # Input layer
    inputs = Input(shape=(32, 32, 3))

    # Block 1: Multiple Convolutional Layers and Pooling
    conv1 = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    avg_pool1 = AveragePooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(conv1)

    conv2 = Conv2D(128, (3, 3), activation='relu', padding='same')(avg_pool1)
    avg_pool2 = AveragePooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(conv2)

    conv3 = Conv2D(256, (3, 3), activation='relu', padding='same')(avg_pool2)
    avg_pool3 = AveragePooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(conv3)

    # Block 2: Concatenation of Features from Different Scales
    concat = Concatenate(axis=-1)([avg_pool3, conv3, conv2, conv1])

    # Flatten and Classify
    flatten = Flatten()(concat)
    dense1 = Dense(128, activation='relu')(flatten)
    dense2 = Dense(10, activation='softmax')(dense1)

    # Construct the model
    model = Model(inputs=inputs, outputs=dense2)

    return model

# Instantiate the model
model = dl_model()
model.summary()