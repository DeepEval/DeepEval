import keras
from keras.layers import Input, Conv2D, Concatenate, Add, BatchNormalization, Flatten, Dense

def dl_model():
    # Define the input layer
    input_layer = Input(shape=(32, 32, 3))

    # Define the first feature extraction path
    path1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Define the second feature extraction path
    path2 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    path2 = Conv2D(filters=32, kernel_size=(1, 7), strides=(1, 1), padding='same', activation='relu')(path2)
    path2 = Conv2D(filters=32, kernel_size=(7, 1), strides=(1, 1), padding='same', activation='relu')(path2)

    # Concatenate the outputs of the two paths
    concatenated_path = Concatenate()([path1, path2])

    # Apply a 1x1 convolution to align the output dimensions
    aligned_path = Conv2D(filters=3, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(concatenated_path)

    # Define a branch that connects directly to the input
    branch = input_layer

    # Merge the outputs of the main path and the branch through addition
    merged_path = Add()([aligned_path, branch])

    # Apply batch normalization
    batch_norm = BatchNormalization()(merged_path)

    # Flatten the output
    flatten_layer = Flatten()(batch_norm)

    # Define the first fully connected layer
    dense1 = Dense(units=128, activation='relu')(flatten_layer)

    # Define the second fully connected layer
    output_layer = Dense(units=10, activation='softmax')(dense1)

    # Create the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model