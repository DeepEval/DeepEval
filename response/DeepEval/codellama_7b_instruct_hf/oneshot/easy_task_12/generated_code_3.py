import keras
from keras.layers import Input, Conv2D, MaxPooling2D, SeparableConv2D, BatchNormalization, Flatten, Dense

def dl_model():
    # Define the input layer with a shape of (28, 28, 1) for the MNIST dataset
    input_layer = Input(shape=(28, 28, 1))

    # Define the main path with two blocks
    main_path = input_layer
    for i in range(2):
        main_path = SeparableConv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(main_path)
        main_path = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(main_path)

    # Define the branch path with a 1x1 convolutional layer
    branch_path = SeparableConv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Sum the outputs from the main and branch paths
    merged_path = keras.layers.concatenate([main_path, branch_path], axis=3)

    # Flatten the merged path
    flattened_path = Flatten()(merged_path)

    # Add a fully connected layer to generate the final probability distribution
    output_layer = Dense(units=10, activation='softmax')(flattened_path)

    # Define the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model