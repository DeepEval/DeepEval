import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Concatenate, Flatten, Dense, BatchNormalization
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 pixels with 3 color channels

    # Path 1: 1x1 Convolution
    path1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Path 2: 1x1 followed by 1x7 and 7x1 Convolutions
    path2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    path2 = Conv2D(filters=64, kernel_size=(1, 7), strides=(1, 1), padding='same', activation='relu')(path2)
    path2 = Conv2D(filters=64, kernel_size=(7, 1), strides=(1, 1), padding='same', activation='relu')(path2)

    # Path 3: 1x1 followed by two sets of 1x7 and 7x1 Convolutions
    path3 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    path3 = Conv2D(filters=64, kernel_size=(1, 7), strides=(1, 1), padding='same', activation='relu')(path3)
    path3 = Conv2D(filters=64, kernel_size=(7, 1), strides=(1, 1), padding='same', activation='relu')(path3)
    path3 = Conv2D(filters=64, kernel_size=(1, 7), strides=(1, 1), padding='same', activation='relu')(path3)
    path3 = Conv2D(filters=64, kernel_size=(7, 1), strides=(1, 1), padding='same', activation='relu')(path3)

    # Path 4: Average Pooling followed by a 1x1 Convolution
    path4 = AveragePooling2D(pool_size=(2, 2), strides=(1, 1), padding='same')(input_layer)
    path4 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(path4)

    # Concatenating all paths
    concatenated = Concatenate()([path1, path2, path3, path4])

    # Batch normalization
    batch_norm = BatchNormalization()(concatenated)

    # Flattening the output
    flatten_layer = Flatten()(batch_norm)

    # Fully connected layer for classification
    dense_layer = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense_layer)

    # Constructing the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model