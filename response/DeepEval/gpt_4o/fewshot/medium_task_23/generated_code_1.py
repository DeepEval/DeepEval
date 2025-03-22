import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Concatenate, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32x3

    # Path 1: Single 1x1 convolution
    path1 = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(input_layer)

    # Path 2: 1x1 convolution followed by 1x7 and 7x1 convolutions
    path2_1 = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(input_layer)
    path2_2 = Conv2D(filters=64, kernel_size=(1, 7), padding='same', activation='relu')(path2_1)
    path2_3 = Conv2D(filters=64, kernel_size=(7, 1), padding='same', activation='relu')(path2_2)

    # Path 3: 1x1 convolution followed by two sets of 1x7 and 7x1 convolutions
    path3_1 = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(input_layer)
    path3_2 = Conv2D(filters=64, kernel_size=(1, 7), padding='same', activation='relu')(path3_1)
    path3_3 = Conv2D(filters=64, kernel_size=(7, 1), padding='same', activation='relu')(path3_2)
    path3_4 = Conv2D(filters=64, kernel_size=(1, 7), padding='same', activation='relu')(path3_3)
    path3_5 = Conv2D(filters=64, kernel_size=(7, 1), padding='same', activation='relu')(path3_4)

    # Path 4: Average pooling followed by a 1x1 convolution
    path4_1 = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(input_layer)
    path4_2 = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(path4_1)

    # Concatenate outputs from all paths
    concatenated = Concatenate()([path1, path2_3, path3_5, path4_2])

    # Flatten and fully connected layer for classification
    flatten = Flatten()(concatenated)
    output_layer = Dense(units=10, activation='softmax')(flatten)  # CIFAR-10 has 10 classes

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model