import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    # Define the input layer
    input_layer = Input(shape=(32, 32, 3))

    # First branch: local features through a 3x3 convolutional layer
    branch1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Second branch: downsampling using a max pooling layer
    branch2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(input_layer)
    branch2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch2)

    # Third branch: upsampling using an upsampling layer
    branch3 = Upsampling2D(size=(2, 2))(branch2)
    branch3 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch3)

    # Fuse the outputs of all branches through concatenation
    concatenated_branches = Concatenate()([branch1, branch2, branch3])

    # Apply batch normalization and flatten the result
    batch_norm = BatchNormalization()(concatenated_branches)
    flattened_branches = Flatten()(batch_norm)

    # Pass the flattened result through three fully connected layers
    dense1 = Dense(units=128, activation='relu')(flattened_branches)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    # Create the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model