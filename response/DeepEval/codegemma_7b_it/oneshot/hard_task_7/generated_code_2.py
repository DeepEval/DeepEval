import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Lambda, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    # Define the input layer
    input_layer = Input(shape=(28, 28, 1))

    # Initial convolutional layer
    conv_init = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(input_layer)

    # Block 1
    def block1(input_tensor):
        # Split the input into two groups
        group1, group2 = tf.split(input_tensor, num_or_size_splits=2, axis=-1)

        # Operations for the first group
        conv1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(group1)
        conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
        conv3 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv2)

        # Operations for the second group
        conv4 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(group2)

        # Concatenate the outputs
        output_tensor = Concatenate()([conv1, conv2, conv3, conv4])

        return output_tensor

    # Block 2
    def block2(input_tensor):
        # Get the shape of the input
        input_shape = keras.backend.int_shape(input_tensor)

        # Reshape the input into four groups
        group_size = input_shape[-1] // 4
        input_tensor = tf.reshape(input_tensor, shape=(-1, input_shape[1], input_shape[2], group_size, 4))

        # Swap the third and fourth dimensions
        input_tensor = tf.transpose(input_tensor, perm=[0, 1, 2, 4, 3])

        # Reshape the input back to the original shape
        input_tensor = tf.reshape(input_tensor, shape=(-1, input_shape[1], input_shape[2], input_shape[-1]))

        # Flatten the input
        input_tensor = Flatten()(input_tensor)

        # Fully connected layer for classification
        output_tensor = Dense(units=10, activation='softmax')(input_tensor)

        return output_tensor

    # Apply the blocks to the input
    block1_output = block1(conv_init)
    block2_output = block2(block1_output)

    # Create the model
    model = keras.Model(inputs=input_layer, outputs=block2_output)

    return model