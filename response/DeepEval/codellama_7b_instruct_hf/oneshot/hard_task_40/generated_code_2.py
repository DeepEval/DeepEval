import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    # First block
    conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    max_pool1 = MaxPooling2D(pool_size=(1, 1), strides=1, padding='same')(conv1)
    flatten1 = Flatten()(max_pool1)

    conv2 = Conv2D(filters=32, kernel_size=(2, 2), strides=(1, 1), padding='same', activation='relu')(input_layer)
    max_pool2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(conv2)
    flatten2 = Flatten()(max_pool2)

    conv3 = Conv2D(filters=32, kernel_size=(4, 4), strides=(1, 1), padding='same', activation='relu')(input_layer)
    max_pool3 = MaxPooling2D(pool_size=(4, 4), strides=4, padding='same')(conv3)
    flatten3 = Flatten()(max_pool3)

    # Concatenate outputs from all pooling layers
    flattened = Concatenate()([flatten1, flatten2, flatten3])

    # Fully connected layers
    dense1 = Dense(units=128, activation='relu')(flattened)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model