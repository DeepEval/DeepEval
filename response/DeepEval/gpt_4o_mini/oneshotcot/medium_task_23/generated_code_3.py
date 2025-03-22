import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Concatenate, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 with 3 color channels

    # Path 1: Single 1x1 convolution
    path1 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(input_layer)

    # Path 2: 1x1 convolution followed by 1x7 and 7x1 convolutions
    path2 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(input_layer)
    path2 = Conv2D(filters=32, kernel_size=(1, 7), padding='same', activation='relu')(path2)
    path2 = Conv2D(filters=32, kernel_size=(7, 1), padding='same', activation='relu')(path2)

    # Path 3: 1x1 convolution followed by two sets of 1x7 and 7x1 convolutions
    path3 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(input_layer)
    path3 = Conv2D(filters=32, kernel_size=(1, 7), padding='same', activation='relu')(path3)
    path3 = Conv2D(filters=32, kernel_size=(7, 1), padding='same', activation='relu')(path3)
    path3 = Conv2D(filters=32, kernel_size=(1, 7), padding='same', activation='relu')(path3)
    path3 = Conv2D(filters=32, kernel_size=(7, 1), padding='same', activation='relu')(path3)

    # Path 4: Average pooling followed by a 1x1 convolution
    path4 = AveragePooling2D(pool_size=(2, 2), padding='same')(input_layer)
    path4 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(path4)

    # Concatenate all paths
    output_tensor = Concatenate()([path1, path2, path3, path4])

    # Flatten the result
    flatten_layer = Flatten()(output_tensor)

    # Fully connected layer for classification
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)  # 10 classes for CIFAR-10

    # Create the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model