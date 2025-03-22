import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))

    # Two parallel convolutional branches with varying kernel sizes
    conv1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv2 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Concatenate the outputs of the two branches
    concatenated = Concatenate()([conv1, conv2])

    # Global average pooling layer to compress features
    pooling = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(concatenated)

    # Two fully connected layers with softmax activation to generate attention weights
    dense1 = Dense(units=128, activation='relu')(pooling)
    dense2 = Dense(units=64, activation='softmax')(dense1)

    # Multiply the output of each branch by its corresponding weight and add to produce the final weighted output
    weighted = Concatenate()([dense2[0] * conv1, dense2[1] * conv2])

    # Fully connected layer to output probability distribution across 10 classes
    final = Dense(units=10, activation='softmax')(weighted)

    # Define the model
    model = keras.Model(inputs=input_layer, outputs=final)

    return model