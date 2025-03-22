import keras
from keras.layers import Input, Conv2D, MaxPooling2D, AveragePooling2D, Concatenate, Flatten, Dense, BatchNormalization
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 pixels with 3 color channels

    # Path 1: 1x1 Convolution
    path1 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(input_layer)

    # Path 2: Average Pooling + 1x1 Convolution
    path2 = AveragePooling2D(pool_size=(2, 2), padding='same')(input_layer)
    path2 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(path2)

    # Path 3: 1x1 Convolution + Parallel 1x3 and 3x1 Convolutions
    path3 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(input_layer)
    path3_1 = Conv2D(filters=32, kernel_size=(1, 3), padding='same', activation='relu')(path3)
    path3_2 = Conv2D(filters=32, kernel_size=(3, 1), padding='same', activation='relu')(path3)
    path3_output = Concatenate()([path3_1, path3_2])

    # Path 4: 1x1 Convolution + 3x3 Convolution + Parallel 1x3 and 3x1 Convolutions
    path4 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(input_layer)
    path4 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(path4)
    path4_1 = Conv2D(filters=32, kernel_size=(1, 3), padding='same', activation='relu')(path4)
    path4_2 = Conv2D(filters=32, kernel_size=(3, 1), padding='same', activation='relu')(path4)
    path4_output = Concatenate()([path4_1, path4_2])

    # Concatenate all paths
    concatenated = Concatenate()([path1, path2, path3_output, path4_output])

    # Batch normalization
    batch_norm = BatchNormalization()(concatenated)

    # Flatten and fully connected layers
    flatten_layer = Flatten()(batch_norm)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)  # 10 classes for CIFAR-10

    # Construct the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model