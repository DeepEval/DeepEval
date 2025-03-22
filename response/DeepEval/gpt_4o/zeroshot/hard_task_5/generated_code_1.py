from tensorflow.keras.layers import Input, Lambda, Conv2D, DepthwiseConv2D, SeparableConv2D, Add, Flatten, Dense
from tensorflow.keras.models import Model
import tensorflow as tf

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))

    # Block 1: Split input into three groups and apply 1x1 convolution
    def block_1(x):
        # Split into three groups
        split = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(x)
        # Apply 1x1 convolution to each group
        convs = [Conv2D(x.shape[-1] // 3, (1, 1), padding='same', activation='relu')(s) for s in split]
        # Concatenate along the channel dimension
        return tf.keras.layers.Concatenate(axis=-1)(convs)

    # Apply Block 1
    block1_out = block_1(input_layer)

    # Block 2: Reshape for channel shuffling
    def block_2(x):
        input_shape = tf.shape(x)
        # Reshape
        reshaped = tf.reshape(x, (input_shape[0], input_shape[1], input_shape[2], 3, -1))
        # Permutation to swap third and fourth dimensions
        permuted = tf.transpose(reshaped, [0, 1, 2, 4, 3])
        # Reshape back to original shape
        return tf.reshape(permuted, input_shape)

    # Apply Block 2
    block2_out = Lambda(block_2)(block1_out)

    # Block 3: Depthwise separable convolution
    block3_out = SeparableConv2D(block2_out.shape[-1], (3, 3), padding='same', activation='relu')(block2_out)

    # Branch: Direct connection to input
    branch = Conv2D(block3_out.shape[-1], (1, 1), padding='same', activation='relu')(input_layer)

    # Combine main path and branch with addition
    combined_out = Add()([block3_out, branch])

    # Flatten and fully connected layer for classification
    flattened = Flatten()(combined_out)
    output_layer = Dense(10, activation='softmax')(flattened)

    # Construct the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model

# Example usage:
# model = dl_model()
# model.summary()