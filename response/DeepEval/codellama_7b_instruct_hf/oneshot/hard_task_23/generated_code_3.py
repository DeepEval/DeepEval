import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Activation
from keras.applications.vgg16 import VGG16

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))

    # Initial Convolutional Layer
    conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='valid', activation='relu')(input_layer)

    # Branch 1: Local Feature Extraction
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
    conv3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)

    # Branch 2 and 3: Average Pooling and Upsampling
    pool1 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv1)
    transconv1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(pool1)

    # Concatenation and Refining
    conv4 = Conv2D(filters=128, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(Concatenate()([conv2, conv3, transconv1]))

    # Fully Connected Layer
    flatten = Flatten()(conv4)
    dense1 = Dense(units=10, activation='softmax')(flatten)

    model = keras.Model(inputs=input_layer, outputs=dense1)

    return model