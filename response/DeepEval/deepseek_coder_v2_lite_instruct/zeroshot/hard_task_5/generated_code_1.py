import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, ReLU, Add, Lambda, Reshape, Permute, DepthwiseConv2D, Flatten, Dense

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))

    # Block 1
    x = Conv2D(32, (1, 1), padding='same')(input_layer)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    # Splitting the input into three groups
    splits = Lambda(lambda tensor: tf.split(tensor, num_or_size_splits=3, axis=-1))(x)

    # Processing each group with a 1x1 convolutional layer
    processed_splits = []
    for split in splits:
        processed = Conv2D(split.shape[-1] // 3, (1, 1), padding='same')(split)
        processed_splits.append(processed)

    # Concatenating the outputs from the three groups
    x = Add()(processed_splits)

    # Block 2
    # Obtaining the shape of the feature from Block 1
    shape = tf.keras.backend.int_shape(x)
    height, width, channels = shape[1], shape[2], shape[3]

    # Reshaping into (height, width, groups, channels_per_group)
    x = Reshape((height, width, 3, channels // 3))(x)

    # Swapping the third and fourth dimensions
    x = Permute((1, 2, 4, 3))(x)

    # Reshaping back to the original shape to achieve channel shuffling
    x = Reshape((height, width, channels))(x)

    # Block 3
    x = DepthwiseConv2D((3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    # Branch connecting directly to the input
    branch = input_layer

    # Combining the outputs from the main path and the branch through an addition operation
    x = Add()([x, branch])

    # Flattening the output
    x = Flatten()(x)

    # Final output through a fully connected layer
    output = Dense(10, activation='softmax')(x)

    # Constructing the model
    model = Model(inputs=input_layer, outputs=output)

    return model

# Example usage:
# model = dl_model()
# model.summary()