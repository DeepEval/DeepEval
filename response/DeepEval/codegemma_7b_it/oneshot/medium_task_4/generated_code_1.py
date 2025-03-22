import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Add

def dl_model():

    input_layer = Input(shape=(32, 32, 3))

    # Path1: Two blocks of convolution followed by average pooling
    path1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    path1 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(path1)
    path1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(path1)
    path1 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(path1)

    # Path2: Single convolutional layer
    path2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    path2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(path2)

    # Concatenate the outputs from both pathways
    concat = Concatenate()([path1, path2])

    # Flatten the concatenated vector
    flatten_layer = Flatten()(concat)

    # Fully connected layer to map to probabilities
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model