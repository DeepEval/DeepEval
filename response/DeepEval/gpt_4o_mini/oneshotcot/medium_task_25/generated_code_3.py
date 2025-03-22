import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 pixels with 3 color channels

    # Path 1: 1x1 convolution
    path1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Path 2: Average pooling followed by 1x1 convolution
    path2 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(input_layer)
    path2 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(path2)

    # Path 3: 1x1 convolution followed by parallel 1x3 and 3x1 convolutions
    path3_1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    path3_2 = Conv2D(filters=32, kernel_size=(1, 3), strides=(1, 1), padding='same', activation='relu')(path3_1)
    path3_3 = Conv2D(filters=32, kernel_size=(3, 1), strides=(1, 1), padding='same', activation='relu')(path3_1)
    path3 = Concatenate()([path3_2, path3_3])

    # Path 4: 1x1 convolution followed by 3x3 convolution, then parallel 1x3 and 3x1 convolutions
    path4_1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    path4_2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(path4_1)
    path4_3 = Conv2D(filters=32, kernel_size=(1, 3), strides=(1, 1), padding='same', activation='relu')(path4_2)
    path4_4 = Conv2D(filters=32, kernel_size=(3, 1), strides=(1, 1), padding='same', activation='relu')(path4_2)
    path4 = Concatenate()([path4_3, path4_4])

    # Concatenate all paths
    concat_paths = Concatenate()([path1, path2, path3, path4])

    # Add Batch Normalization
    batch_norm = BatchNormalization()(concat_paths)

    # Flatten the output
    flatten_layer = Flatten()(batch_norm)

    # Fully connected layer for classification
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)  # CIFAR-10 has 10 classes

    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model