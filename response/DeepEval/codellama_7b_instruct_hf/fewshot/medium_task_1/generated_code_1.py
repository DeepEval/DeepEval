import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Add, Flatten, Dense


def dl_model():
    # Define the input layer
    input_layer = Input(shape=(32, 32, 3))

    # Define the first convolutional layer
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Define the second convolutional layer
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)

    # Define the max-pooling layer
    pool = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv2)

    # Define the addition layer
    add = Add()([input_layer, pool])

    # Define the flatten layer
    flatten = Flatten()(add)

    # Define the first fully connected layer
    fc1 = Dense(units=128, activation='relu')(flatten)

    # Define the second fully connected layer
    fc2 = Dense(units=10, activation='softmax')(fc1)

    # Create the model
    model = keras.Model(inputs=input_layer, outputs=fc2)