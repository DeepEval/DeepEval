import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Flatten, Concatenate, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Path 1: 1x1 Convolution
    path1 = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(input_layer)

    # Path 2: 1x1 Convolution followed by 1x7 and 7x1 Convolutions
    path2_conv1 = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(input_layer)
    path2_conv2 = Conv2D(filters=64, kernel_size=(1, 7), padding='same', activation='relu')(path2_conv1)
    path2 = Conv2D(filters=64, kernel_size=(7, 1), padding='same', activation='relu')(path2_conv2)

    # Path 3: 1x1 Convolution followed by two sets of 1x7 and 7x1 Convolutions
    path3_conv1 = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(input_layer)
    path3_conv2 = Conv2D(filters=64, kernel_size=(1, 7), padding='same', activation='relu')(path3_conv1)
    path3_conv3 = Conv2D(filters=64, kernel_size=(7, 1), padding='same', activation='relu')(path3_conv2)
    path3_conv4 = Conv2D(filters=64, kernel_size=(1, 7), padding='same', activation='relu')(path3_conv3)
    path3 = Conv2D(filters=64, kernel_size=(7, 1), padding='same', activation='relu')(path3_conv4)

    # Path 4: Average Pooling followed by 1x1 Convolution
    path4_pool = AveragePooling2D(pool_size=(2, 2), strides=2, padding='same')(input_layer)
    path4 = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(path4_pool)

    # Concatenate the outputs of all paths
    concatenated = Concatenate()([path1, path2, path3, path4])

    # Flatten the result
    flatten = Flatten()(concatenated)

    # Fully connected layer for classification
    output_layer = Dense(units=10, activation='softmax')(flatten)

    # Create model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model