import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense

def dl_model():
    # Input layer
    input_layer = Input(shape=(28, 28, 1))

    # First block
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    maxpool1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv1)
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(maxpool1)
    maxpool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv2)
    conv3 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(maxpool2)
    maxpool3 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv3)

    # Second block
    flatten_layer = Flatten()(maxpool3)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    # Create model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model