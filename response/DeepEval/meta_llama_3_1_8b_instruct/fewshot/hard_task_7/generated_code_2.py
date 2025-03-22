import keras
import tensorflow as tf
from keras.layers import Input, Conv2D, Lambda, Concatenate, DepthwiseConv2D, Reshape, Permute, Flatten, Dense

def dl_model():

    input_layer = Input(shape=(28,28,1))

    # Convolutional layer to adjust the dimensions of the input data
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Block 1
    def block_1(input_tensor):
        # Split the input into two groups along the last dimension
        inputs_groups = Lambda(lambda x: tf.split(value=x, num_or_size_splits=2, axis=-1))(input_tensor)

        # Process the first group
        conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(inputs_groups[0])
        depthwise_conv = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
        conv2 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(depthwise_conv)
        output_group1 = Concatenate()([conv1, depthwise_conv, conv2])

        # Pass the second group without modification
        output_group2 = inputs_groups[1]

        # Merge the outputs from both groups
        output_tensor = Concatenate()([output_group1, output_group2])
        return output_tensor

    block1_output = block_1(input_tensor=conv1)

    # Block 2
    def block_2(input_tensor):
        # Get the shape of the input
        input_shape = Lambda(lambda x: tf.shape(x))(input_tensor)
        height = input_shape[1]
        width = input_shape[2]
        channels = input_shape[3]

        # Reshape the input into four groups
        reshaped = Reshape(target_shape=(height, width, -1, channels//4))(input_tensor)

        # Swap the third and fourth dimensions
        permuted = Permute((1, 2, 4, 3))(reshaped)

        # Reshape the input back to its original shape to achieve channel shuffling
        reshaped_back = Reshape(target_shape=(height, width, channels))(permuted)

        return reshaped_back

    block2_output = block_2(input_tensor=block1_output)

    # Flatten the output and pass it through a fully connected layer for classification
    flatten = Flatten()(block2_output)
    output_layer = Dense(units=10, activation='softmax')(flatten)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model

def main():
    model = dl_model()
    model.summary()
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

if __name__ == "__main__":
    main()