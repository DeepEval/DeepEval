import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():

    # Define the input layer
    input_layer = Input(shape=(28, 28, 1))

    # Define the first processing pathway
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
    conv3 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv2)
    pool1 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(conv3)
    batch_norm1 = BatchNormalization()(pool1)

    # Define the second processing pathway
    conv4 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv5 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv4)
    conv6 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv5)
    pool2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(conv6)
    batch_norm2 = BatchNormalization()(pool2)

    # Concatenate the outputs from both pathways
    merged = Concatenate()([batch_norm1, batch_norm2])

    # Apply a 3x3 convolution to the concatenated output
    conv7 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(merged)

    # Flatten the output and add two fully connected layers
    flatten = Flatten()(conv7)
    dense1 = Dense(units=128, activation='relu')(flatten)
    dense2 = Dense(units=10, activation='softmax')(dense1)

    # Define the model
    model = keras.Model(inputs=input_layer, outputs=dense2)

    return model