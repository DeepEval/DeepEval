import keras
from keras.layers import Input, Conv2D, Concatenate, Flatten, Dense
from keras.datasets import cifar10

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 with 3 color channels

    # First convolutional layer
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(input_layer)
    concat1 = Concatenate(axis=-1)([input_layer, conv1])  # Concatenating input and conv1 output

    # Second convolutional layer
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(concat1)
    concat2 = Concatenate(axis=-1)([concat1, conv2])  # Concatenating concat1 and conv2 output

    # Third convolutional layer
    conv3 = Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu')(concat2)
    concat3 = Concatenate(axis=-1)([concat2, conv3])  # Concatenating concat2 and conv3 output

    # Flattening and fully connected layers
    flatten_layer = Flatten()(concat3)
    dense1 = Dense(units=256, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)  # 10 units for 10 classes in CIFAR-10

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model