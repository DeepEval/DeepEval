import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense

def dl_model():
    # Input layer
    input_layer = Input(shape=(28, 28, 1))

    # Convolutional and Max Pooling layers
    conv_layer = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    max_pooling_layer = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(conv_layer)

    # Second Convolutional layer
    conv_layer_2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(max_pooling_layer)

    # Flatten layer
    flatten_layer = Flatten()(conv_layer_2)

    # Fully connected layers
    dense_layer_1 = Dense(units=128, activation='relu')(flatten_layer)
    dense_layer_2 = Dense(units=10, activation='softmax')(dense_layer_1)

    # Define the model
    model = keras.Model(inputs=input_layer, outputs=dense_layer_2)

    return model