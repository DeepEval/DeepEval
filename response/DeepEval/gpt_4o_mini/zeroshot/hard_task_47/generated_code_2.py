import tensorflow as tf
from tensorflow.keras import layers, models

def dl_model():
    input_shape = (32, 32, 3)  # CIFAR-10 images are 32x32 pixels with 3 color channels
    inputs = layers.Input(shape=input_shape)

    # First block: Split input and apply depthwise separable convolutions
    split_outputs = layers.Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(inputs)
    
    depthwise_conv_outputs = []
    kernel_sizes = [1, 3, 5]  # Kernel sizes for depthwise separable convolutions

    for i, kernel_size in enumerate(kernel_sizes):
        x = layers.SeparableConv2D(filters=32, kernel_size=(kernel_size, kernel_size), padding='same')(split_outputs[i])
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        depthwise_conv_outputs.append(x)

    # Concatenate outputs of the first block
    block1_output = layers.Concatenate()(depthwise_conv_outputs)

    # Second block: Multiple branches for feature extraction
    branch1 = layers.Conv2D(filters=32, kernel_size=(1, 1), padding='same')(block1_output)
    branch1 = layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same')(branch1)

    branch2 = layers.Conv2D(filters=32, kernel_size=(1, 1), padding='same')(block1_output)
    branch2 = layers.Conv2D(filters=64, kernel_size=(1, 7), padding='same')(branch2)
    branch2 = layers.Conv2D(filters=64, kernel_size=(7, 1), padding='same')(branch2)
    branch2 = layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same')(branch2)

    branch3 = layers.AveragePooling2D(pool_size=(2, 2))(block1_output)

    # Concatenate outputs of the second block
    block2_output = layers.Concatenate()([branch1, branch2, branch3])

    # Final classification layers
    x = layers.GlobalAveragePooling2D()(block2_output)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.5)(x)  # Optional dropout for regularization
    outputs = layers.Dense(10, activation='softmax')(x)  # 10 classes for CIFAR-10

    # Construct the model
    model = models.Model(inputs=inputs, outputs=outputs)

    return model

# Example of how to create the model
model = dl_model()
model.summary()  # To display the model architecture