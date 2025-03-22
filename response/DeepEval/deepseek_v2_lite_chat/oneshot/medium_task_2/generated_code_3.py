import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    # Define the input layer
    input_layer = Input(shape=(32, 32, 3))

    # Define the main path
    def main_path(input_tensor):
        conv1 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(input_tensor)
        conv2 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(conv1)
        pool = MaxPooling2D(pool_size=(2, 2), padding='same')(conv2)

        return pool

    # Define the branch path
    def branch_path(input_tensor):
        conv = Conv2D(filters=64, kernel_size=(5, 5), activation='relu', padding='same')(input_tensor)
        return conv

    # Apply the paths
    main_pool = main_path(input_layer)
    branch_conv = branch_path(input_layer)

    # Concatenate the outputs of both paths
    concatenated_tensor = Concatenate(axis=-1)([main_pool, branch_conv])

    # Flatten the concatenated tensor
    flattened_tensor = Flatten()(concatenated_tensor)

    # Map the flattened tensor to a probability distribution
    dense1 = Dense(units=512, activation='relu')(flattened_tensor)
    dense2 = Dense(units=256, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model