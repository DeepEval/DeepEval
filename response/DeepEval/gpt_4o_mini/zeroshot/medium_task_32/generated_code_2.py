import tensorflow as tf
from tensorflow.keras import layers, models

def dl_model():
    # Input layer
    inputs = layers.Input(shape=(32, 32, 3))

    # Split the input into three groups along the last dimension
    split = layers.Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(inputs)

    # Apply depthwise separable convolution for each group with different kernel sizes
    conv1 = layers.SeparableConv2D(32, kernel_size=(1, 1), activation='relu')(split[0])
    conv2 = layers.SeparableConv2D(32, kernel_size=(3, 3), activation='relu')(split[1])
    conv3 = layers.SeparableConv2D(32, kernel_size=(5, 5), activation='relu')(split[2])

    # Concatenate the outputs of the three convolutional layers
    concatenated = layers.Concatenate()([conv1, conv2, conv3])

    # Flatten the concatenated features
    flattened = layers.Flatten()(concatenated)

    # Fully connected layer for classification
    outputs = layers.Dense(10, activation='softmax')(flattened)  # 10 classes for CIFAR-10

    # Construct the model
    model = models.Model(inputs=inputs, outputs=outputs)

    return model

# Example of creating the model
model = dl_model()
model.summary()