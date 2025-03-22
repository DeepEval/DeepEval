import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))

    # Path 1: 1x1 convolution
    path1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Path 2: 1x1 convolution followed by 1x7 and 7x1 convolutions
    path2 = Conv2D(filters=64, kernel_size=(1, 7), strides=(1, 1), padding='same', activation='relu')(input_layer)
    path2 = Conv2D(filters=64, kernel_size=(7, 1), strides=(1, 1), padding='same', activation='relu')(path2)

    # Path 3: 1x1 convolution followed by a combination of two sets of 1x7 and 7x1 convolutions to capture multi-directional spatial features
    path3 = Conv2D(filters=64, kernel_size=(1, 7), strides=(1, 1), padding='same', activation='relu')(input_layer)
    path3 = Conv2D(filters=64, kernel_size=(7, 1), strides=(1, 1), padding='same', activation='relu')(path3)
    path3 = Conv2D(filters=64, kernel_size=(1, 7), strides=(1, 1), padding='same', activation='relu')(path3)
    path3 = Conv2D(filters=64, kernel_size=(7, 1), strides=(1, 1), padding='same', activation='relu')(path3)

    # Path 4: average pooling followed by a 1x1 convolution
    path4 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(input_layer)
    path4 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(path4)

    # Fuse outputs of paths through concatenation
    output_tensor = Concatenate()([path1, path2, path3, path4])

    # Add batch normalization and flatten the output
    bath_norm = BatchNormalization()(output_tensor)
    flatten_layer = Flatten()(bath_norm)

    # Add fully connected layers for classification
    dense1 = Dense(units=64, activation='relu')(flatten_layer)
    dense2 = Dense(units=32, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    # Create and return the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    return model