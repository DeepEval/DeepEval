import tensorflow as tf
from tensorflow.keras import layers, models

def dl_model(input_shape=(32, 32, 3), num_classes=10):
    inputs = layers.Input(shape=input_shape)

    # Block 1
    # Split input into three groups
    split_inputs = layers.Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(inputs)

    # Process each group with 1x1 convolution
    conv1 = layers.Conv2D(filters=input_shape[-1] // 3, kernel_size=(1, 1), activation='relu')(split_inputs[0])
    conv2 = layers.Conv2D(filters=input_shape[-1] // 3, kernel_size=(1, 1), activation='relu')(split_inputs[1])
    conv3 = layers.Conv2D(filters=input_shape[-1] // 3, kernel_size=(1, 1), activation='relu')(split_inputs[2])

    # Concatenate along the channel dimension
    block1_output = layers.Concatenate(axis=-1)([conv1, conv2, conv3])

    # Block 2
    # Obtain shape of the output from Block 1
    height, width, channels = block1_output.shape[1:]

    # Reshape into (height, width, groups, channels_per_group)
    reshaped = layers.Reshape((height, width, 3, channels // 3))(block1_output)
    
    # Permute dimensions to achieve channel shuffling
    permuted = layers.Permute((0, 1, 3, 2))(reshaped)

    # Reshape back to the original shape
    block2_output = layers.Reshape((height, width, channels))(permuted)

    # Block 3
    # Apply 3x3 depthwise separable convolution
    depthwise_conv = layers.SeparableConv2D(filters=channels, kernel_size=(3, 3), padding='same', activation='relu')(block2_output)

    # Branch from the input
    branch_output = layers.Conv2D(filters=channels, kernel_size=(1, 1), padding='same', activation='relu')(inputs)

    # Combine main path and branch output
    combined_output = layers.Add()([depthwise_conv, branch_output])

    # Final fully connected layer
    flatten_output = layers.Flatten()(combined_output)
    final_output = layers.Dense(num_classes, activation='softmax')(flatten_output)

    # Construct the model
    model = models.Model(inputs=inputs, outputs=final_output)

    return model

# Example of how to use the model
model = dl_model()
model.summary()