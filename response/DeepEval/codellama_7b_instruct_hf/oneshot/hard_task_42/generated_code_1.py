import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    # Input layer
    input_layer = Input(shape=(28, 28, 1))

    # Block 1
    # Three parallel paths with max pooling layers of different scales
    path1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    path2 = Conv2D(filters=32, kernel_size=(2, 2), strides=(1, 1), padding='same', activation='relu')(input_layer)
    path3 = Conv2D(filters=32, kernel_size=(4, 4), strides=(1, 1), padding='same', activation='relu')(input_layer)
    path1 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(path1)
    path2 = MaxPooling2D(pool_size=(4, 4), strides=4, padding='same')(path2)
    path3 = MaxPooling2D(pool_size=(8, 8), strides=8, padding='same')(path3)

    # Flatten and concatenate pooling results
    path1_flat = Flatten()(path1)
    path2_flat = Flatten()(path2)
    path3_flat = Flatten()(path3)
    concat_layer = Concatenate()([path1_flat, path2_flat, path3_flat])

    # Batch normalization and flatten output
    batch_norm = BatchNormalization()(concat_layer)
    flat_layer = Flatten()(batch_norm)

    # Block 2
    # Four parallel paths with convolution and pooling layers
    path1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(flat_layer)
    path2 = Conv2D(filters=64, kernel_size=(1, 7), strides=(1, 1), padding='same', activation='relu')(flat_layer)
    path3 = Conv2D(filters=64, kernel_size=(7, 1), strides=(1, 1), padding='same', activation='relu')(flat_layer)
    path4 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(flat_layer)
    path1 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(path1)
    path2 = MaxPooling2D(pool_size=(4, 4), strides=4, padding='same')(path2)
    path3 = MaxPooling2D(pool_size=(8, 8), strides=8, padding='same')(path3)
    path4 = MaxPooling2D(pool_size=(16, 16), strides=16, padding='same')(path4)

    # Concatenate output of all paths
    output_layer = Concatenate()([path1, path2, path3, path4])

    # Fully connected layers and output
    dense1 = Dense(units=128, activation='relu')(output_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    # Create and return model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model