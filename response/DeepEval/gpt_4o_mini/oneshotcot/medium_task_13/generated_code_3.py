import keras
from keras.layers import Input, Conv2D, Concatenate, Flatten, Dense
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 with 3 color channels

    # First convolutional layer
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(input_layer)
    # Concatenate input with output of first layer
    concat1 = Concatenate()([input_layer, conv1])

    # Second convolutional layer
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(concat1)
    # Concatenate input with output of second layer
    concat2 = Concatenate()([concat1, conv2])

    # Third convolutional layer
    conv3 = Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu')(concat2)

    # Final concatenation of the last convolution output with the previous layers
    final_concat = Concatenate()([concat2, conv3])

    # Flatten the final concatenated output
    flatten_layer = Flatten()(final_concat)

    # Fully connected layers
    dense1 = Dense(units=256, activation='relu')(flatten_layer)
    dense2 = Dense(units=10, activation='softmax')(dense1)  # 10 classes for CIFAR-10

    # Construct the model
    model = Model(inputs=input_layer, outputs=dense2)

    return model