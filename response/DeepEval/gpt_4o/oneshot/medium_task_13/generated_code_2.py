import keras
from keras.layers import Input, Conv2D, Concatenate, Flatten, Dense
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # First convolutional layer
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(input_layer)

    # Concatenate input and first convolutional layer output
    concat1 = Concatenate(axis=-1)([input_layer, conv1])

    # Second convolutional layer
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(concat1)

    # Concatenate first concat output and second convolutional layer output
    concat2 = Concatenate(axis=-1)([concat1, conv2])

    # Third convolutional layer
    conv3 = Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu')(concat2)

    # Concatenate second concat output and third convolutional layer output
    concat3 = Concatenate(axis=-1)([concat2, conv3])

    # Flatten the output
    flatten_layer = Flatten()(concat3)

    # Fully connected layers
    dense1 = Dense(units=256, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    # Define the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model