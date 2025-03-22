import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))

    # Reshape input tensor into groups
    groups = 3
    channels_per_group = 3
    reshaped_input = keras.layers.Reshape(target_shape=(32, 32, groups, channels_per_group))(input_layer)

    # Swap third and fourth dimensions for channel shuffling
    swapped_input = keras.layers.Permute((0, 1, 3, 2))(reshaped_input)

    # Reshape back to original input shape
    final_input = keras.layers.Reshape(target_shape=(32, 32, 3))(swapped_input)

    # Convolutional layers
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(final_input)
    conv2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(conv2)

    # Batch normalization and flattening
    batch_norm = BatchNormalization()(pool1)
    flat = Flatten()(batch_norm)

    # Fully connected layers
    dense1 = Dense(units=128, activation='relu')(flat)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    # Create model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model