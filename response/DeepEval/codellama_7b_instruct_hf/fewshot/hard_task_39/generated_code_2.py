import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Concatenate
from keras.models import Model

def dl_model():

    # Define the input layer
    input_layer = Input(shape=(28, 28, 1))

    # Define the first block
    conv1_1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    maxpool1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv1_1)
    flatten1 = Flatten()(maxpool1)
    conv1_2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(maxpool1)
    maxpool2 = MaxPooling2D(pool_size=(4, 4), strides=(4, 4), padding='same')(conv1_2)
    flatten2 = Flatten()(maxpool2)
    conv1_3 = Conv2D(filters=128, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(maxpool2)
    maxpool3 = MaxPooling2D(pool_size=(8, 8), strides=(8, 8), padding='same')(conv1_3)
    flatten3 = Flatten()(maxpool3)

    # Define the second block
    conv2_1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(flatten1)
    conv2_2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(flatten2)
    conv2_3 = Conv2D(filters=128, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(flatten3)
    maxpool2 = MaxPooling2D(pool_size=(8, 8), strides=(8, 8), padding='same')(conv2_3)

    # Concatenate the outputs from all branches
    concatenated = Concatenate()([conv2_1, conv2_2, conv2_3, maxpool2])

    # Flatten the concatenated output
    flattened = Flatten()(concatenated)

    # Add a fully connected layer and a softmax activation function
    dense = Dense(units=10, activation='softmax')(flattened)

    # Create the model
    model = Model(inputs=input_layer, outputs=dense)

    return model