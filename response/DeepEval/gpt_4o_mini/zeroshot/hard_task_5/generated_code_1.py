import tensorflow as tf
from tensorflow.keras import layers, models

def dl_model():
    input_shape = (32, 32, 3)  # CIFAR-10 image size
    inputs = layers.Input(shape=input_shape)

    # Block 1: Split input into three groups and process with 1x1 conv
    split_inputs = layers.Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(inputs)
    conv1 = layers.Conv2D(filters=input_shape[-1] // 3, kernel_size=(1, 1), activation='relu')(split_inputs[0])
    conv2 = layers.Conv2D(filters=input_shape[-1] // 3, kernel_size=(1, 1), activation='relu')(split_inputs[1])
    conv3 = layers.Conv2D(filters=input_shape[-1] // 3, kernel_size=(1, 1), activation='relu')(split_inputs[2])
    
    fused_features = layers.Concatenate(axis=-1)([conv1, conv2, conv3])

    # Block 2: Channel shuffle
    batch_size, height, width, channels = tf.shape(fused_features)[0], tf.shape(fused_features)[1], tf.shape(fused_features)[2], tf.shape(fused_features)[3]
    reshaped = layers.Reshape((height, width, 3, channels // 3))(fused_features)
    shuffled = layers.Permute((1, 2, 4, 3))(reshaped)
    shuffled_back = layers.Reshape((height, width, channels))(shuffled)

    # Block 3: Depthwise separable convolution
    depthwise_conv = layers.SeparableConv2D(filters=channels, kernel_size=(3, 3), padding='same', activation='relu')(shuffled_back)

    # Branch from input
    branch_output = layers.Conv2D(filters=channels, kernel_size=(1, 1), padding='same', activation='relu')(inputs)

    # Combine the outputs from the main path and the branch
    combined_output = layers.Add()([depthwise_conv, branch_output])

    # Fully connected layer
    flatten = layers.Flatten()(combined_output)
    outputs = layers.Dense(10, activation='softmax')(flatten)  # 10 classes for CIFAR-10

    # Create the model
    model = models.Model(inputs=inputs, outputs=outputs)

    return model

# Example of how to create the model
model = dl_model()
model.summary()