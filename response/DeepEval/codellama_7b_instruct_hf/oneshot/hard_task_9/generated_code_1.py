import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():

    # Define the input layer
    input_layer = Input(shape=(32, 32, 3))

    # Define the first feature extraction branch
    branch1 = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    branch1 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(branch1)

    # Define the second feature extraction branch
    branch2 = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    branch2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch2)
    branch2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(branch2)

    # Define the third feature extraction branch
    branch3 = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    branch3 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch3)
    branch3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch3)
    branch3 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(branch3)

    # Concatenate the outputs from the three branches
    x = Concatenate()([branch1, branch2, branch3])

    # Adjust the output dimensions to match the input image's channel size
    x = Conv2D(filters=3, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(x)

    # Apply batch normalization and flatten the result
    x = BatchNormalization()(x)
    x = Flatten()(x)

    # Output the classification result through three fully connected layers
    x = Dense(units=128, activation='relu')(x)
    x = Dense(units=64, activation='relu')(x)
    x = Dense(units=10, activation='softmax')(x)

    # Define the model
    model = keras.Model(inputs=input_layer, outputs=x)

    return model