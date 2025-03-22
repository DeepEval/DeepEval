import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Add, Flatten, Dense

def dl_model():     

    input_layer = Input(shape=(28, 28, 1))

    # Block 1
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    pool1 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(conv1)

    # Block 2
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(pool1)
    pool2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(conv2)

    # Combine outputs
    combined_features = Add()([pool2, conv1])

    # Flatten and classify
    flatten_layer = Flatten()(combined_features)
    dense_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=dense_layer)

    return model