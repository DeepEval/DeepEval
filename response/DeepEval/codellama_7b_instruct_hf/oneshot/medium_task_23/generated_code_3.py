import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Path 1: Single 1x1 convolution
    path1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Path 2: 1x1 convolution followed by 1x7 and 7x1 convolutions
    path2 = Conv2D(filters=32, kernel_size=(1, 7), strides=(1, 1), padding='same', activation='relu')(input_layer)
    path2 = Conv2D(filters=32, kernel_size=(7, 1), strides=(1, 1), padding='same', activation='relu')(path2)

    # Path 3: 1x1 convolution followed by a combination of two sets of 1x7 and 7x1 convolutions
    path3 = Conv2D(filters=32, kernel_size=(1, 7), strides=(1, 1), padding='same', activation='relu')(input_layer)
    path3 = Conv2D(filters=32, kernel_size=(7, 1), strides=(1, 1), padding='same', activation='relu')(path3)
    path3 = Conv2D(filters=32, kernel_size=(1, 7), strides=(1, 1), padding='same', activation='relu')(path3)
    path3 = Conv2D(filters=32, kernel_size=(7, 1), strides=(1, 1), padding='same', activation='relu')(path3)

    # Path 4: Average pooling followed by a 1x1 convolution
    path4 = MaxPooling2D(pool_size=(2, 2), strides=2)(input_layer)
    path4 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(path4)

    # Fuse the outputs of the paths through concatenation
    merged_path = Concatenate()([path1, path2, path3, path4])

    # Apply batch normalization and flatten the result
    batch_norm = BatchNormalization()(merged_path)
    flattened = Flatten()(batch_norm)

    # Pass the flattened result through a fully connected layer for classification
    output = Dense(units=10, activation='softmax')(flattened)

    # Define the model
    model = keras.Model(inputs=input_layer, outputs=output)

    return model