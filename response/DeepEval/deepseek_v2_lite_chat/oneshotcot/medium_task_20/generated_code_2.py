import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Average

def dl_model():
    # Define the input shape (CIFAR-10 images are 3x32x32)
    input_layer = Input(shape=(32, 32, 3))

    # First path
    path1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Second path with two 3x3 convolutions
    path2 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(path1)
    path2 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(path2)

    # Third path with a single 3x3 convolution
    path3 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(input_layer)

    # Fourth path with max pooling followed by a 1x1 convolution
    path4 = MaxPooling2D(pool_size=(2, 2))(input_layer)
    path4 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(path4)

    # Concatenate the outputs of the paths
    concatenated = Concatenate()(list(path1) + list(path2) + list(path3) + list(path4))

    # Flatten the concatenated tensor
    flattened = Flatten()(concatenated)

    # Dense layer with 128 units
    dense1 = Dense(units=128, activation='relu')(flattened)

    # Final dense layer with softmax activation
    output_layer = Dense(units=10, activation='softmax')(dense1)

    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model

# Instantiate and return the constructed model
model = dl_model()