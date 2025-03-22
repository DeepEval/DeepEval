import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    # Define the input layer
    input_layer = Input(shape=(28, 28, 1))

    # Block 1
    # 3 parallel paths with different pooling sizes
    path1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(2, 2), padding='same', activation='relu')(input_layer)
    path2 = Conv2D(filters=32, kernel_size=(2, 2), strides=(4, 4), padding='same', activation='relu')(input_layer)
    path3 = Conv2D(filters=32, kernel_size=(4, 4), strides=(8, 8), padding='same', activation='relu')(input_layer)

    # Flatten and concatenate the outputs
    flattened_output = Flatten()(Concatenate()([path1, path2, path3]))

    # Apply batch normalization
    batch_norm = BatchNormalization()(flattened_output)

    # Block 2
    # 3 feature extraction branches
    branch1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(batch_norm)
    branch2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(batch_norm)
    branch3 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(batch_norm)

    # Concatenate the outputs of the 3 branches
    concatenated_output = Concatenate()([branch1, branch2, branch3])

    # Flatten and concatenate the output of the concatenation
    flattened_output = Flatten()(concatenated_output)

    # Apply a fully connected layer and a reshape operation
    dense_layer = Dense(units=64, activation='relu')(flattened_output)
    reshaped_output = Reshape((4, 4, 64))(dense_layer)

    # Apply another fully connected layer and a reshape operation
    dense_layer = Dense(units=64, activation='relu')(reshaped_output)
    reshaped_output = Reshape((4, 4, 64))(dense_layer)

    # Apply a final fully connected layer for classification
    output_layer = Dense(units=10, activation='softmax')(reshaped_output)

    # Create the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model