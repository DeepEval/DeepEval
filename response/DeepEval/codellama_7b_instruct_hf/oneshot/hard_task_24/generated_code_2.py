import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    # Define the input layer
    input_layer = Input(shape=(32, 32, 3))

    # First branch: 1x1 convolutional layer
    conv1 = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Second branch: 3x3 convolutional layer
    conv2 = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Third branch: max pooling layer for downsampling
    max_pooling = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(conv1)

    # Fourth branch: 3x3 convolutional layer
    conv3 = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(max_pooling)

    # Fifth branch: upsampling layer
    up_sampling = UpSampling2D(size=(2, 2))(conv3)

    # Sixth branch: 1x1 convolutional layer
    conv4 = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(up_sampling)

    # Fuse the outputs of all branches
    concatenate = Concatenate()([conv1, conv2, conv3, conv4])

    # Apply batch normalization and flatten the result
    batch_norm = BatchNormalization()(concatenate)
    flatten = Flatten()(batch_norm)

    # Pass the flattened output through three fully connected layers to produce the final classification
    dense1 = Dense(units=128, activation='relu')(flatten)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    # Define the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model