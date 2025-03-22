import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Lambda, Permute, DepthwiseConv2D
from tensorflow.keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Main path starts here
    def block1(input_tensor):
        # Split the input into 3 groups
        split_1 = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_tensor)
        split_2 = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_tensor)
        split_3 = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_tensor)

        # Process each group with a 1x1 convolutional layer
        processed_1 = Conv2D(filters=input_tensor.shape[-1]//3, kernel_size=(1, 1), activation='relu')(split_1)
        processed_2 = Conv2D(filters=input_tensor.shape[-1]//3, kernel_size=(1, 1), activation='relu')(split_2)
        processed_3 = Conv2D(filters=input_tensor.shape[-1]//3, kernel_size=(1, 1), activation='relu')(split_3)

        # Concatenate the processed outputs
        concatenated = Concatenate(axis=-1)([processed_1, processed_2, processed_3])
        return concatenated

    # Block 2
    def block2(input_tensor):
        # Get the shape of the input tensor
        shape = input_tensor.shape
        # Reshape into groups
        reshaped = tf.reshape(input_tensor, shape=(shape[0], shape[1], shape[3], 3))
        # Swap the third and fourth dimensions
        permuted = Permute((1, 2, 4, 3))(reshaped)
        # Reshape back to original shape
        shuffled = tf.reshape(permuted, shape=shape)
        return shuffled

    # Block 3
    def block3(input_tensor):
        depthwise = DepthwiseConv2D(kernel_size=(3, 3), padding='same')(input_tensor)
        return depthwise

    # Apply blocks to the input layer
    main_path_output = block1(input_tensor=input_layer)
    main_path_output = block2(input_tensor=main_path_output)
    main_path_output = block3(input_tensor=main_path_output)

    # Branch that connects directly to the input
    branch_output = input_layer

    # Combine the outputs from the main path and the branch
    combined_output = tf.add(main_path_output, branch_output)

    # Flatten the combined output
    flattened = Flatten()(combined_output)

    # Fully connected layers
    dense1 = Dense(units=256, activation='relu')(flattened)
    dense2 = Dense(units=128, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    # Construct the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model

# Example usage
model = dl_model()
model.summary()