import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    # Define the input layer
    input_layer = Input(shape=(32, 32, 3))

    # Define the first block
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
    maxpool = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(conv2)
    block1 = Concatenate()([conv1, conv2, maxpool])

    # Define the second block
    conv3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(block1)
    conv4 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv3)
    maxpool = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(conv4)
    block2 = Concatenate()([conv3, conv4, maxpool])

    # Define the third block
    conv5 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(block2)
    conv6 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv5)
    maxpool = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(conv6)
    block3 = Concatenate()([conv5, conv6, maxpool])

    # Flatten the output of the third block
    flattened = Flatten()(block3)

    # Define the fully connected layers
    dense1 = Dense(units=128, activation='relu')(flattened)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    # Define the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model