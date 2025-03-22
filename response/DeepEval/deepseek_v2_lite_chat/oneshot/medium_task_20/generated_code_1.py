import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))

    # First convolutional path
    conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    maxpool1 = MaxPooling2D(pool_size=(2, 2), strides=1, padding='valid')(conv1)

    # Second convolutional path
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(maxpool1)
    maxpool2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv2)

    # Third convolutional path
    conv3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(maxpool2)
    maxpool3 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv3)

    # Fourth convolutional path
    conv4 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(maxpool3)
    maxpool4 = MaxPooling2D(pool_size=(1, 1), strides=2, padding='valid')(conv4)

    # Concatenate the outputs of the four paths
    concat_layer = Concatenate(axis=-1)([conv1, conv2, conv3, maxpool4])

    # Batch normalization and flattening
    batch_norm = BatchNormalization()(concat_layer)
    flatten_layer = Flatten()(batch_norm)

    # Dense layer
    dense1 = Dense(units=128, activation='relu')(flatten_layer)

    # Output layer
    output_layer = Dense(units=10, activation='softmax')(dense1)

    # Model construction
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model

# Instantiate and return the constructed model
model = dl_model()