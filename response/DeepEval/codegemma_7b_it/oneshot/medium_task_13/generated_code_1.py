import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    # Input layer for CIFAR-10 dataset
    input_layer = Input(shape=(32, 32, 3))

    # First convolutional layer
    conv1 = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(input_layer)

    # Second convolutional layer
    conv2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(conv1)

    # Third convolutional layer
    conv3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(conv2)

    # Concatenate outputs of all convolutional layers
    concat = Concatenate()([conv1, conv2, conv3])

    # Flatten the concatenated output
    flatten_layer = Flatten()(concat)

    # Fully connected layer
    dense1 = Dense(units=128, activation='relu')(flatten_layer)

    # Output layer for multi-class classification
    output_layer = Dense(units=10, activation='softmax')(dense1)

    # Create the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model