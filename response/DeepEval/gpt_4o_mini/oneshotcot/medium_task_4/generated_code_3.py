import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Flatten, Dense, Add
from keras.models import Model

def dl_model():     
    # Input layer
    input_layer = Input(shape=(32, 32, 3))

    # Path 1: Two blocks of convolution followed by average pooling
    path1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    path1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(path1)
    path1 = AveragePooling2D(pool_size=(2, 2), strides=2, padding='valid')(path1)

    path1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(path1)
    path1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(path1)
    path1 = AveragePooling2D(pool_size=(2, 2), strides=2, padding='valid')(path1)

    # Path 2: Single convolutional layer
    path2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Combine the paths through addition
    combined = Add()([path1, path2])

    # Flatten the combined output
    flatten_layer = Flatten()(combined)

    # Fully connected layer for classification
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    # Create the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model