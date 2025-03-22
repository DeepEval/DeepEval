from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras import backend as K

def dl_model():

    input_layer = Input(shape=(28, 28, 1))

    # First convolution layer
    conv_1 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(input_layer)

    # Feature extraction using 3x3 convolution layer
    conv_2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(conv_1)

    # Second convolution layer to restore dimensionality
    conv_3 = Conv2D(filters=16, kernel_size=(1, 1), padding='same', activation='relu')(conv_2)

    # Flattening the output
    flatten = Flatten()(conv_3)

    # Fully connected layer for classification
    dense = Dense(units=10, activation='softmax')(flatten)

    # Creating the model
    model = keras.Model(inputs=input_layer, outputs=dense)

    return model

model = dl_model()