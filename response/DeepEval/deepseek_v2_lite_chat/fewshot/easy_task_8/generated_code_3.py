import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    # Depthwise separable convolutional layer
    dw_conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    dw_conv = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(dw_conv)

    # Max pooling layer
    pool = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(dw_conv)

    # Dropout layer for regularization
    pool = Dropout(0.25)(pool)

    # Flatten layer
    flat = Flatten()(pool)

    # Fully connected layer
    dense = Dense(units=128, activation='relu')(flat)

    # Output layer for classification
    output_layer = Dense(units=10, activation='softmax')(dense)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model