import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Concatenate, Flatten, Dense, BatchNormalization
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 with 3 color channels

    # Path 1: 1x1 Convolution
    path1 = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(input_layer)

    # Path 2: Average Pooling followed by 1x1 Convolution
    path2 = AveragePooling2D(pool_size=(2, 2), strides=(1, 1), padding='same')(input_layer)
    path2 = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(path2)

    # Path 3: 1x1 Convolution followed by two parallel convolutions (1x3 and 3x1)
    path3 = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(input_layer)
    path3_1 = Conv2D(filters=64, kernel_size=(1, 3), padding='same', activation='relu')(path3)
    path3_2 = Conv2D(filters=64, kernel_size=(3, 1), padding='same', activation='relu')(path3)
    path3 = Concatenate()([path3_1, path3_2])

    # Path 4: 1x1 Convolution followed by 3x3 Convolution, then parallel convolutions (1x3 and 3x1)
    path4 = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(input_layer)
    path4 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(path4)
    path4_1 = Conv2D(filters=64, kernel_size=(1, 3), padding='same', activation='relu')(path4)
    path4_2 = Conv2D(filters=64, kernel_size=(3, 1), padding='same', activation='relu')(path4)
    path4 = Concatenate()([path4_1, path4_2])

    # Concatenate all paths
    concatenated = Concatenate()([path1, path2, path3, path4])

    # Batch Normalization and Flatten
    batch_norm = BatchNormalization()(concatenated)
    flatten_layer = Flatten()(batch_norm)

    # Fully Connected Layer for classification
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)  # 10 classes for CIFAR-10

    model = Model(inputs=input_layer, outputs=output_layer)

    return model