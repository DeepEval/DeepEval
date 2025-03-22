import tensorflow as tf
from tensorflow.keras import layers, models

def dl_model():
    # Input layer for MNIST images (28x28x1)
    input_shape = (28, 28, 1)
    inputs = layers.Input(shape=input_shape)

    # First block - Three average pooling layers with different window sizes
    pooling1 = layers.AveragePooling2D(pool_size=(1, 1), strides=(1, 1))(inputs)
    pooling2 = layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2))(inputs)
    pooling3 = layers.AveragePooling2D(pool_size=(4, 4), strides=(4, 4))(inputs)

    # Flatten each pooling output
    flat1 = layers.Flatten()(pooling1)
    flat2 = layers.Flatten()(pooling2)
    flat3 = layers.Flatten()(pooling3)

    # Concatenate the flattened outputs
    concatenated = layers.concatenate([flat1, flat2, flat3])

    # Fully connected layer
    dense1 = layers.Dense(128, activation='relu')(concatenated)

    # Reshape the output into 4D tensor (batch_size, height, width, channels)
    reshaped = layers.Reshape((1, 1, 128))(dense1)  # Reshaping to (1, 1, 128)

    # Second block - Split the input into four groups and apply depthwise separable convolutions
    split = layers.Lambda(lambda x: tf.split(x, num_or_size_splits=4, axis=-1))(reshaped)

    # Define different kernel sizes for depthwise separable convolutions
    depthwise_outputs = []
    kernel_sizes = [1, 3, 5, 7]
    
    for kernel_size in kernel_sizes:
        depthwise_conv = layers.SeparableConv2D(filters=32, kernel_size=(kernel_size, kernel_size), padding='same', activation='relu')(split.pop(0))
        depthwise_outputs.append(depthwise_conv)

    # Concatenate the outputs from the depthwise separable convolutions
    concatenated_depthwise = layers.concatenate(depthwise_outputs)

    # Flatten the concatenated output
    flatten_depthwise = layers.Flatten()(concatenated_depthwise)

    # Fully connected layer for classification
    output = layers.Dense(10, activation='softmax')(flatten_depthwise)  # 10 classes for MNIST

    # Create the model
    model = models.Model(inputs=inputs, outputs=output)

    return model

# You can instantiate the model
model = dl_model()
model.summary()