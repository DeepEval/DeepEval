import keras
from keras.layers import Input, Dense, GlobalAveragePooling2D, Flatten, Concatenate
from keras.models import Model


def dl_model():
    # Define the input shape
    input_shape = (32, 32, 3)

    # Define the first branch
    branch_1 = Input(shape=input_shape)

    # Apply global average pooling to compress the input features
    branch_1 = GlobalAveragePooling2D()(branch_1)

    # Pass the pooled output through two fully connected layers to produce weights whose dimension is equal to the channel of input
    branch_1 = Dense(64, activation='relu')(branch_1)
    branch_1 = Dense(16, activation='relu')(branch_1)

    # Reshape the weights to match the input's shape
    branch_1 = Flatten()(branch_1)
    branch_1 = Reshape(input_shape[1:])(branch_1)

    # Element-wise multiply the weights with the input
    branch_1 = branch_1 * branch_1

    # Define the second branch
    branch_2 = Input(shape=input_shape)

    # Apply global average pooling to compress the input features
    branch_2 = GlobalAveragePooling2D()(branch_2)

    # Pass the pooled output through two fully connected layers to produce weights whose dimension is equal to the channel of input
    branch_2 = Dense(64, activation='relu')(branch_2)
    branch_2 = Dense(16, activation='relu')(branch_2)

    # Reshape the weights to match the input's shape
    branch_2 = Flatten()(branch_2)
    branch_2 = Reshape(input_shape[1:])(branch_2)

    # Element-wise multiply the weights with the input
    branch_2 = branch_2 * branch_2

    # Concatenate the outputs from both branches
    concatenated = Concatenate()([branch_1, branch_2])

    # Flatten the output
    flattened = Flatten()(concatenated)

    # Pass the flattened output through a fully connected layer to produce the final classification
    output = Dense(10, activation='softmax')(flattened)

    # Define the model
    model = Model(inputs=[branch_1, branch_2], outputs=output)

    return model