import tensorflow as tf
from tensorflow.keras import layers, models, Input

def dl_model():
    input_shape = (32, 32, 3)  # CIFAR-10 images are 32x32 pixels with 3 color channels (RGB)
    inputs = Input(shape=input_shape)

    # Block 1
    split_inputs = layers.Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(inputs)

    # Feature extraction with separable convolutions
    def separable_conv_block(x, kernel_size):
        x = layers.SeparableConv2D(32, kernel_size=kernel_size, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        return x

    conv1 = separable_conv_block(split_inputs[0], (1, 1))
    conv2 = separable_conv_block(split_inputs[1], (3, 3))
    conv3 = separable_conv_block(split_inputs[2], (5, 5))

    # Concatenate outputs from Block 1
    block1_output = layers.Concatenate()([conv1, conv2, conv3])

    # Block 2 with four parallel branches
    path1 = layers.Conv2D(32, (1, 1), padding='same')(block1_output)

    path2 = layers.AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(block1_output)
    path2 = layers.Conv2D(32, (1, 1), padding='same')(path2)

    path3 = layers.Conv2D(32, (1, 1), padding='same')(block1_output)
    path3_1 = layers.Conv2D(32, (1, 3), padding='same')(path3)
    path3_2 = layers.Conv2D(32, (3, 1), padding='same')(path3)
    path3 = layers.Concatenate()([path3_1, path3_2])

    path4 = layers.Conv2D(32, (1, 1), padding='same')(block1_output)
    path4 = layers.Conv2D(32, (3, 3), padding='same')(path4)
    path4_1 = layers.Conv2D(32, (1, 3), padding='same')(path4)
    path4_2 = layers.Conv2D(32, (3, 1), padding='same')(path4)
    path4 = layers.Concatenate()([path4_1, path4_2])

    # Concatenate outputs from Block 2
    block2_output = layers.Concatenate()([path1, path2, path3, path4])

    # Final classification layers
    x = layers.Flatten()(block2_output)
    x = layers.Dense(64, activation='relu')(x)
    outputs = layers.Dense(10, activation='softmax')(x)  # 10 classes for CIFAR-10

    # Construct the model
    model = models.Model(inputs=inputs, outputs=outputs)

    return model

# Example usage
if __name__ == "__main__":
    model = dl_model()
    model.summary()