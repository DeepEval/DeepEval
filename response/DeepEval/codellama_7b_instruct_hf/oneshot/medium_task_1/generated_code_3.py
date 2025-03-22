import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
    maxpool = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(conv2)

    # Add the output of the convolutional layers to the input layer
    added_layer = Concatenate()([maxpool, input_layer])

    # Flatten the output of the added layer
    flattened_layer = Flatten()(added_layer)

    # Process the flattened output through two fully connected layers
    dense1 = Dense(units=128, activation='relu')(flattened_layer)
    dense2 = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=dense2)

    return model