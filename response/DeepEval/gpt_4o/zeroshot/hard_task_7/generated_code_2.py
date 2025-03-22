import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, DepthwiseConv2D, SeparableConv2D, Lambda, Concatenate, Reshape, Permute, Flatten, Dense
from tensorflow.keras.models import Model

def dl_model():
    input_shape = (28, 28, 1)  # MNIST images are 28x28 with a single channel
    num_classes = 10  # MNIST has 10 classes for digits 0-9

    inputs = Input(shape=input_shape)

    # Initial convolutional layer
    x = Conv2D(32, (3, 3), padding='same', activation='relu')(inputs)

    # Block 1
    def block_1(x):
        # Split into two groups along the last dimension
        split0, split1 = Lambda(lambda x: tf.split(x, num_or_size_splits=2, axis=-1))(x)

        # First group operations
        group1 = Conv2D(16, (1, 1), padding='same', activation='relu')(split0)
        group1 = SeparableConv2D(16, (3, 3), padding='same', activation='relu')(group1)
        group1 = Conv2D(16, (1, 1), padding='same', activation='relu')(group1)

        # Concatenate the processed first group with the unmodified second group
        merged = Concatenate()([group1, split1])
        return merged

    x = block_1(x)

    # Block 2
    def block_2(x):
        input_shape = tf.shape(x)

        # Reshape to (height, width, groups, channels_per_group)
        height, width, channels = input_shape[1], input_shape[2], input_shape[3]
        groups = 4
        channels_per_group = channels // groups

        reshaped = Reshape((height, width, groups, channels_per_group))(x)

        # Permute to swap the third and fourth dimensions
        shuffled = Permute((1, 2, 4, 3))(reshaped)

        # Reshape back to the original shape
        reshaped_back = Reshape((height, width, channels))(shuffled)

        return reshaped_back

    x = block_2(x)

    # Fully connected layer for classification
    x = Flatten()(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs, outputs)
    return model

# Construct the model
model = dl_model()
model.summary()