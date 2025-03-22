import keras
from keras.layers import Input, Conv2D, Concatenate, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # First convolutional layer
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(input_layer)
    concat1 = Concatenate(axis=-1)([input_layer, conv1])  # Concatenate along the channel dimension

    # Second convolutional layer
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(concat1)
    concat2 = Concatenate(axis=-1)([concat1, conv2])  # Concatenate along the channel dimension

    # Third convolutional layer
    conv3 = Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu')(concat2)

    # Flatten the output
    flatten = Flatten()(conv3)

    # Fully connected layers for classification
    dense1 = Dense(units=256, activation='relu')(flatten)
    output_layer = Dense(units=10, activation='softmax')(dense1)  # 10 classes for CIFAR-10

    # Define the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model