import keras
from keras.layers import Input, Conv2D, Concatenate, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images have shape 32x32 with 3 color channels

    # First convolutional layer
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(input_layer)
    concat1 = Concatenate()([input_layer, conv1])  # Concatenate with input

    # Second convolutional layer
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(concat1)
    concat2 = Concatenate()([concat1, conv2])  # Concatenate with previous output

    # Third convolutional layer
    conv3 = Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu')(concat2)
    concat3 = Concatenate()([concat2, conv3])  # Concatenate with previous output

    # Flatten the output for the fully connected layers
    flatten_layer = Flatten()(concat3)

    # First fully connected layer
    dense1 = Dense(units=256, activation='relu')(flatten_layer)

    # Second fully connected layer
    output_layer = Dense(units=10, activation='softmax')(dense1)  # 10 classes for CIFAR-10

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model