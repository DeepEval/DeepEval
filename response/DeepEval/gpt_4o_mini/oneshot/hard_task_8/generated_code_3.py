import keras
from keras.layers import Input, Conv2D, Dense, Concatenate, Reshape, Permute, Activation
from keras.layers import DepthwiseConv2D, Flatten

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    # Block 1
    # Primary path
    primary_path = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(input_layer)
    primary_path = DepthwiseConv2D(kernel_size=(3, 3), padding='same', activation='relu')(primary_path)
    primary_path = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(primary_path)

    # Branch path
    branch_path = DepthwiseConv2D(kernel_size=(3, 3), padding='same', activation='relu')(input_layer)
    branch_path = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(branch_path)

    # Concatenate the outputs of both paths
    block1_output = Concatenate(axis=-1)([primary_path, branch_path])

    # Block 2
    # Reshape the output of Block 1
    block1_shape = block1_output.shape
    groups = 4
    channels_per_group = block1_shape[-1] // groups
    reshaped_output = Reshape((block1_shape[1], block1_shape[2], groups, channels_per_group))(block1_output)

    # Permute dimensions to achieve channel shuffling
    permuted_output = Permute((0, 1, 3, 2, 4))(reshaped_output)

    # Reshape back to the original shape
    final_output = Reshape((block1_shape[1], block1_shape[2], block1_shape[-1]))(permuted_output)

    # Flatten the output and add a fully connected layer for classification
    flatten_layer = Flatten()(final_output)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model

# Example of creating the model
model = dl_model()
model.summary()  # To check the architecture of the model