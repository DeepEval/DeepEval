import keras
from keras.layers import Input, Conv2D, MaxPooling2D, BatchNormalization, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Depthwise separable convolutional layer
    conv = Conv2D(filters=16, kernel_size=(7, 7), strides=(2, 2), padding='same', activation='relu')(input_layer)
    batch_norm = BatchNormalization()(conv)
    pool = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(batch_norm)

    # Two fully connected layers with the same number of channels as the input layer
    dense1 = Dense(units=16, activation='relu')(pool)
    dense2 = Dense(units=16, activation='relu')(dense1)

    # Add the original input with the processed features
    add = keras.layers.Add()([input_layer, dense2])

    # Flatten the output and pass through the final fully connected layer
    flat = Flatten()(add)
    output = Dense(units=10, activation='softmax')(flat)

    model = keras.Model(inputs=input_layer, outputs=output)
    return model