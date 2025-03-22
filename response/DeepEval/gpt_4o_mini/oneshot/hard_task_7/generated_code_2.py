import keras
from keras.layers import Input, Conv2D, Lambda, DepthwiseConv2D, Concatenate, Flatten, Dense, Reshape
import tensorflow as tf

def dl_model():
    input_layer = Input(shape=(28, 28, 1))
    
    # Initial convolutional layer
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Block 1
    def block1(input_tensor):
        # Split the input tensor into two groups along the last dimension
        split_tensors = Lambda(lambda x: tf.split(x, num_or_size_splits=2, axis=-1))(input_tensor)

        # First group operations
        path1 = Conv2D(filters=16, kernel_size=(1, 1), padding='same', activation='relu')(split_tensors[0])
        path1 = DepthwiseConv2D(kernel_size=(3, 3), padding='same', activation='relu')(path1)
        path1 = Conv2D(filters=16, kernel_size=(1, 1), padding='same', activation='relu')(path1)

        # Second group (unchanged)
        path2 = split_tensors[1]

        # Concatenate both paths
        output_tensor = Concatenate()([path1, path2])

        return output_tensor

    block1_output = block1(input_tensor=conv1)

    # Block 2
    def block2(input_tensor):
        # Reshape input into four groups
        shape = tf.shape(input_tensor)
        channels = shape[-1]
        groups = 4
        channels_per_group = channels // groups

        reshaped_tensor = Reshape((shape[1], shape[2], groups, channels_per_group))(input_tensor)
        permuted_tensor = Lambda(lambda x: tf.transpose(x, perm=[0, 1, 3, 2, 4]))(reshaped_tensor)
        shuffled_tensor = Reshape((shape[1], shape[2], channels))(permuted_tensor)

        return shuffled_tensor

    block2_output = block2(input_tensor=block1_output)

    # Final layers
    flatten_layer = Flatten()(block2_output)
    dense_layer = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model