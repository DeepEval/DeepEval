import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, AveragePooling2D

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))

    # Path 1: Single 1x1 convolution
    path1 = Conv2D(filters=32, kernel_size=(1, 1), padding='same')(input_layer)

    # Path 2: 1x1 convolution followed by 1x7 and 7x1 convolutions
    path2 = Conv2D(filters=32, kernel_size=(1, 1))(input_layer)
    path2 = Conv2D(filters=32, kernel_size=(7, 1))(path2)
    path2 = Conv2D(filters=32, kernel_size=(1, 7))(path2)

    # Path 3: 1x1 convolution followed by a combination of two sets of 1x7 and 7x1 convolutions
    path3 = Conv2D(filters=32, kernel_size=(1, 1))(input_layer)
    path3 = Conv2D(filters=32, kernel_size=(7, 1))(path3)
    path3 = Conv2D(filters=32, kernel_size=(7, 1))(path3)

    # Path 4: Average pooling followed by a 1x1 convolution
    path4 = AveragePooling2D(pool_size=(1, 1))(input_layer)
    path4 = Conv2D(filters=32, kernel_size=(1, 1))(path4)

    # Concatenate the outputs of these paths
    concatenated = Concatenate()(
        [path1, path2, path3, path4])

    # Batch normalization and flattening
    batch_norm = BatchNormalization()(concatenated)
    flattened = Flatten()(batch_norm)

    # Fully connected layers
    dense1 = Dense(units=128, activation='relu')(flattened)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model