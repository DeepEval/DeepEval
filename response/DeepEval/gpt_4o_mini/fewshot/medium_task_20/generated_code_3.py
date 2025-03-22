import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, Flatten, Dense

def dl_model():
    
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images have a shape of 32x32 with 3 color channels

    # Path 1: 1x1 Convolution
    path1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Path 2: 1x1 Convolution followed by two 3x3 Convolutions
    path2 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    path2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(path2)
    path2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(path2)

    # Path 3: 1x1 Convolution followed by a single 3x3 Convolution
    path3 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    path3 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(path3)

    # Path 4: Max Pooling followed by a 1x1 Convolution
    path4 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(input_layer)
    path4 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(path4)

    # Concatenate outputs from all paths
    concatenated = Concatenate()([path1, path2, path3, path4])

    # Flatten the concatenated output
    flatten_layer = Flatten()(concatenated)

    # Fully connected layer with 128 units
    dense_layer = Dense(units=128, activation='relu')(flatten_layer)

    # Final output layer with softmax activation for 10 classes
    output_layer = Dense(units=10, activation='softmax')(dense_layer)

    # Create the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model