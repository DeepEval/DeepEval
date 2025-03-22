import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Add, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # First convolutional block
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv2)

    # Add the output of the convolutional layers to the original input
    added = Add()([pool1, input_layer])

    # Flatten the combined features
    flattened = Flatten()(added)

    # Fully connected layers
    fc1 = Dense(units=128, activation='relu')(flattened)
    output_layer = Dense(units=10, activation='softmax')(fc1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model