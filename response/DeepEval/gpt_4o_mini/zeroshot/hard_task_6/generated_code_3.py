import tensorflow as tf
from tensorflow.keras import layers, models, Input

def dl_model():
    input_shape = (32, 32, 3)  # CIFAR-10 image shape
    inputs = Input(shape=input_shape)

    # Block 1
    def block1(x):
        # Split the input into 3 groups
        splits = layers.Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(x)
        convs = [layers.Conv2D(filters=input_shape[-1] // 3, kernel_size=(1, 1), activation='relu')(s) for s in splits]
        return layers.Concatenate()(convs)

    x = block1(inputs)

    # Block 2
    def block2(x):
        shape = tf.shape(x)
        # Reshape to (batch_size, height, width, groups, channels_per_group)
        reshaped = layers.Reshape((shape[1], shape[2], 3, shape[-1] // 3))(x)
        # Permute dimensions to achieve channel shuffle
        permuted = layers.Permute((1, 2, 4, 3))(reshaped)
        # Reshape back to original shape
        return layers.Reshape(shape[1:])(permuted)

    x = block2(x)

    # Block 3
    def block3(x):
        # Apply 3x3 depthwise separable convolution
        return layers.SeparableConv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(x)

    x = block3(x)

    # Repeat Block 1
    x = block1(x)

    # Branch Path - Average Pooling
    branch = layers.AveragePooling2D(pool_size=(4, 4))(inputs)
    branch = layers.Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(branch)

    # Concatenate Main Path and Branch Path
    x = layers.Concatenate()([x, branch])

    # Fully Connected Layer
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)
    outputs = layers.Dense(10, activation='softmax')(x)  # CIFAR-10 has 10 classes

    # Create the model
    model = models.Model(inputs=inputs, outputs=outputs)

    return model

# Example of how to create the model
model = dl_model()
model.summary()