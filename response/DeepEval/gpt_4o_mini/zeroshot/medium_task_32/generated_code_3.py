import tensorflow as tf
from tensorflow.keras import layers, models

def dl_model():
    # Input layer for CIFAR-10 images
    input_shape = (32, 32, 3)
    inputs = layers.Input(shape=input_shape)

    # Lambda layer to split the input into three groups along the last dimension
    split_inputs = layers.Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(inputs)

    # Define depthwise separable convolution layers with different kernel sizes
    def depthwise_separable_conv(x, kernel_size):
        x = layers.DepthwiseConv2D(kernel_size=kernel_size, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Conv2D(filters=32, kernel_size=(1, 1), padding='same')(x)  # Pointwise convolution
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        return x

    # Apply depthwise separable convolutions with different kernel sizes
    conv1 = depthwise_separable_conv(split_inputs[0], kernel_size=(1, 1))
    conv2 = depthwise_separable_conv(split_inputs[1], kernel_size=(3, 3))
    conv3 = depthwise_separable_conv(split_inputs[2], kernel_size=(5, 5))

    # Concatenate the outputs from the three groups
    concatenated = layers.Concatenate(axis=-1)([conv1, conv2, conv3])

    # Flatten the concatenated features
    flattened = layers.Flatten()(concatenated)

    # Fully connected layer for classification
    outputs = layers.Dense(units=10, activation='softmax')(flattened)  # 10 classes for CIFAR-10

    # Create the model
    model = models.Model(inputs=inputs, outputs=outputs)

    return model

# To create the model, you can call dl_model()
model = dl_model()
model.summary()  # Display the model summary