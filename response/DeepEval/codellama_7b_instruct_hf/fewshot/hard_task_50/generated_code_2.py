import keras
from keras.layers import Input, MaxPooling2D, Flatten, Dense, Dropout, Concatenate, Lambda, DepthwiseConv2D, Reshape

def dl_model():
    # Define input layer
    input_layer = Input(shape=(32, 32, 3))

    # Define first block
    def block_1(input_tensor):
        # Max pooling layers with different scales
        maxpool1 = MaxPooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(input_tensor)
        maxpool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(input_tensor)
        maxpool3 = MaxPooling2D(pool_size=(4, 4), strides=(4, 4), padding='same')(input_tensor)

        # Flatten and concatenate pooling outputs
        flatten1 = Flatten()(maxpool1)
        flatten2 = Flatten()(maxpool2)
        flatten3 = Flatten()(maxpool3)
        output_tensor = Concatenate()([flatten1, flatten2, flatten3])

        # Apply dropout
        dropout_layer = Dropout(rate=0.5)(output_tensor)

        return dropout_layer

    # Define second block
    def block_2(input_tensor):
        # Split input into four groups
        inputs_groups = Lambda(lambda x: tf.split(value=x, num_or_size_splits=4, axis=-1))(input_tensor)

        # Separable convolutional layers with varying kernel sizes
        conv1 = DepthwiseConv2D(kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(inputs_groups[0])
        conv2 = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(inputs_groups[1])
        conv3 = DepthwiseConv2D(kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(inputs_groups[2])
        conv4 = DepthwiseConv2D(kernel_size=(7, 7), strides=(1, 1), padding='same', activation='relu')(inputs_groups[3])

        # Concatenate separable convolutional outputs
        output_tensor = Concatenate()([conv1, conv2, conv3, conv4])

        return output_tensor

    # Apply first block and second block to input layer
    block1_output = block_1(input_layer)
    block2_output = block_2(block1_output)

    # Flatten and pass through fully connected layer
    flatten_layer = Flatten()(block2_output)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    # Define model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model