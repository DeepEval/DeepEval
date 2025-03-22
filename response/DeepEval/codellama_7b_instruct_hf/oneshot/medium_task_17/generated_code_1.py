import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    # Define input layer with shape (height, width, channels)
    input_layer = Input(shape=(32, 32, 3))

    # Reshape input tensor into (height, width, groups, channels_per_group)
    reshaped_input = Input(shape=(32, 32, 3, 1))

    # Swap third and fourth dimensions using a permutation operation
    permuted_input = keras.backend.permute_dimensions(reshaped_input, (0, 1, 3, 2))

    # Reshape back to original input shape
    final_input = keras.backend.reshape(permuted_input, (32, 32, 3))

    # Define convolutional layers
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(final_input)
    max_pooling = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv1)

    # Define fully connected layers
    dense1 = Dense(units=128, activation='relu')(max_pooling)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    # Define model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model