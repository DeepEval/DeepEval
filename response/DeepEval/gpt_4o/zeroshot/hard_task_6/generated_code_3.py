import tensorflow as tf
from tensorflow.keras.layers import Input, Lambda, Conv2D, Activation, AveragePooling2D, Flatten, Dense, Concatenate, DepthwiseConv2D, Reshape, Permute
from tensorflow.keras.models import Model

def dl_model():
    # Input layer for CIFAR-10 images (32x32x3)
    inputs = Input(shape=(32, 32, 3))

    # Block 1
    def block1(x):
        # Split input into three groups
        splits = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(x)
        # Process each split with a 1x1 convolution followed by an activation
        convs = [Conv2D(x.shape[-1] // 3, (1, 1), activation='relu')(split) for split in splits]
        # Concatenate the outputs to form the fused features
        return Concatenate()(convs)

    # Block 2
    def block2(x):
        # Obtain shape information
        h, w, c = x.shape[1], x.shape[2], x.shape[3]
        groups = 3
        channels_per_group = c // groups

        # Reshape to (h, w, groups, channels_per_group)
        x = Reshape((h, w, groups, channels_per_group))(x)
        # Permute to (h, w, channels_per_group, groups)
        x = Permute((1, 2, 4, 3))(x)
        # Reshape back to original shape (h, w, c)
        x = Reshape((h, w, c))(x)

        return x

    # Block 3
    def block3(x):
        # Apply a 3x3 depthwise separable convolution
        x = DepthwiseConv2D((3, 3), padding='same', activation='relu')(x)
        return x

    # Main Path
    x = block1(inputs)
    x = block2(x)
    x = block3(x)
    x = block1(x)

    # Branch Path
    branch = AveragePooling2D(pool_size=(8, 8))(inputs)
    branch = Flatten()(branch)

    # Concatenate Main and Branch Path
    x = Flatten()(x)
    x = Concatenate()([x, branch])

    # Fully Connected Layer for Classification
    x = Dense(10, activation='softmax')(x)

    # Create Model
    model = Model(inputs=inputs, outputs=x)

    return model

# Example of how to create the model
model = dl_model()
model.summary()