import keras
from keras.models import Model
from keras.layers import Input, Dense, Dropout
from keras.applications.mobilenet import MobileNet
from keras.layers.convolutional import DepthwiseSeparableConv2D

def dl_model():
    # Define the input shape
    input_shape = (28, 28, 1)

    # Define the depthwise separable convolutional block
    def depthwise_separable_block(x):
        x = DepthwiseSeparableConv2D(32, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(0.2)(x)
        return x

    # Define the 1x1 convolutional block
    def one_by_one_block(x):
        x = Conv2D(16, (1, 1), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(0.2)(x)
        return x

    # Define the model
    model = Sequential()

    # Add the input layer
    model.add(InputLayer(input_shape))

    # Add the depthwise separable convolutional block
    model.add(depthwise_separable_block)

    # Add the 1x1 convolutional block
    model.add(one_by_one_block)

    # Add the concatenate layer
    model.add(Concatenate())

    # Add the fully connected layers
    model.add(Dense(64, activation='relu'))
    model.add(Dense(10))

    # Add the output layer
    model.add(Dense(10, activation='softmax'))

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model