import tensorflow as tf
from tensorflow.keras import layers, models

def dl_model():
    # Input layer
    input_shape = (32, 32, 3)  # CIFAR-10 images are 32x32 RGB images
    inputs = layers.Input(shape=input_shape)

    # Block 1: Global Average Pooling and Fully Connected Layers
    x1 = layers.GlobalAveragePooling2D()(inputs)  # Global Average Pooling
    x1 = layers.Dense(32, activation='relu')(x1)   # First fully connected layer
    x1 = layers.Dense(32, activation='relu')(x1)   # Second fully connected layer
    x1 = layers.Reshape((1, 1, 32))(x1)            # Reshape to (1, 1, 32)

    # Multiply with input to produce weighted feature output
    weighted_features = layers.multiply([inputs, x1])  # Element-wise multiplication

    # Block 2: Convolutional Layers and Max Pooling
    x2 = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(weighted_features)
    x2 = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(x2)
    x2 = layers.MaxPooling2D(pool_size=(2, 2))(x2)  # Max Pooling layer

    # Combine outputs from Block 1 and Block 2
    combined = layers.add([x2, weighted_features])  # Element-wise addition

    # Classification head
    x = layers.GlobalAveragePooling2D()(combined)  # Global Average Pooling
    x = layers.Dense(64, activation='relu')(x)     # First fully connected layer
    outputs = layers.Dense(10, activation='softmax')(x)  # Output layer for classification

    # Create model
    model = models.Model(inputs=inputs, outputs=outputs)

    return model

# Example of creating the model
model = dl_model()
model.summary()  # Print the model summary to check the architecture