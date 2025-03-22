import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    # First convolutional block
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(input_layer)
    conv2 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(conv1)
    max_pool1 = MaxPooling2D(pool_size=(2, 2))(conv2)

    # Second convolutional block
    conv3 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(max_pool1)
    conv4 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(conv3)
    max_pool2 = MaxPooling2D(pool_size=(2, 2))(conv4)

    # Third convolutional block
    conv5 = Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same')(max_pool2)
    conv6 = Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same')(conv5)
    conv7 = Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same')(conv6)
    max_pool3 = MaxPooling2D(pool_size=(2, 2))(conv7)

    # Fourth convolutional block
    conv8 = Conv2D(filters=256, kernel_size=(3, 3), activation='relu', padding='same')(max_pool3)
    conv9 = Conv2D(filters=256, kernel_size=(3, 3), activation='relu', padding='same')(conv8)
    conv10 = Conv2D(filters=256, kernel_size=(3, 3), activation='relu', padding='same')(conv9)
    max_pool4 = MaxPooling2D(pool_size=(2, 2))(conv10)

    # Flattening and fully connected layers
    flatten_layer = Flatten()(max_pool4)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model