import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # 1x1 Convolutions
    conv1_1 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(input_layer)
    conv2_1 = Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(conv1_1)

    # 3x3 Convolutions
    conv1_3 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(input_layer)
    conv2_3 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(conv1_3)

    # 5x5 Convolutions
    conv1_5 = Conv2D(filters=32, kernel_size=(5, 5), activation='relu', padding='same')(input_layer)
    conv2_5 = Conv2D(filters=64, kernel_size=(5, 5), activation='relu', padding='same')(conv1_5)

    # Max Pooling
    pool = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(conv2_3)

    # Concatenate Feature Maps
    combined_features = Concatenate()([conv2_1, conv2_5, pool])

    # Flatten and Fully Connected Layers
    flatten = Flatten()(combined_features)
    dense1 = Dense(units=128, activation='relu')(flatten)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model