import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, Flatten, Dense
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 pixels with 3 color channels

    # Defining the four parallel paths
    # Path 1: 1x1 Convolution
    path1 = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(input_layer)

    # Path 2: 1x1 Convolution followed by two 3x3 Convolutions
    path2 = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(input_layer)
    path2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(path2)
    path2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(path2)

    # Path 3: 1x1 Convolution followed by a single 3x3 Convolution
    path3 = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(input_layer)
    path3 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(path3)

    # Path 4: Max Pooling followed by a 1x1 Convolution
    path4 = MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same')(input_layer)
    path4 = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(path4)

    # Concatenate the outputs of the four paths
    concatenated_output = Concatenate()([path1, path2, path3, path4])

    # Flatten the concatenated output
    flatten_layer = Flatten()(concatenated_output)

    # Dense layer with 128 units
    dense_layer = Dense(units=128, activation='relu')(flatten_layer)

    # Output layer with 10 units (for CIFAR-10 classes) and softmax activation
    output_layer = Dense(units=10, activation='softmax')(dense_layer)

    # Construct the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model