import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, Flatten, Dense

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))

    # First convolutional layer
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    # Concatenate along the channel dimension
    concat1 = Concatenate(axis=-1)([conv1, input_layer])

    # Second convolutional layer
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(concat1)
    # Concatenate along the channel dimension
    concat2 = Concatenate(axis=-1)([conv2, concat1])

    # Third convolutional layer
    conv3 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(concat2)
    # Concatenate along the channel dimension
    concat3 = Concatenate(axis=-1)([conv3, concat2])

    # Flatten the output
    flatten_layer = Flatten()(concat3)

    # Fully connected layers
    dense1 = Dense(units=256, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model