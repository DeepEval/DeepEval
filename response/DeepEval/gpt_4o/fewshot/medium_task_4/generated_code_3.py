import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Add, Flatten, Dense

def dl_model():

    input_layer = Input(shape=(32, 32, 3))

    # Path1: Two blocks of convolution followed by average pooling
    conv1_1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    pool1 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv1_1)
    conv1_2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(pool1)
    path1 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv1_2)

    # Path2: A single convolutional layer
    path2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Combine both paths
    combined = Add()([path1, path2])

    # Flatten and fully connected layer
    flatten_layer = Flatten()(combined)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model