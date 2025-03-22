import tensorflow as tf
from tensorflow.keras import layers, models

def dl_model():
    # Input layer for MNIST images (28x28 grayscale images)
    input_layer = layers.Input(shape=(28, 28, 1))

    # Main path
    main_path = input_layer
    for _ in range(3):  # Repeat the block 3 times
        # Convolutional layer followed by ReLU activation
        x = layers.SeparableConv2D(32, kernel_size=(3, 3), padding='same', activation='relu')(main_path)
        x = layers.BatchNormalization()(x)
        main_path = layers.Concatenate()([main_path, x])  # Concatenate along channel dimension

    # Branch path
    branch_path = layers.Conv2D(32, kernel_size=(1, 1), padding='same')(main_path)

    # Fusion of main and branch paths
    fused = layers.Add()([main_path, branch_path])

    # Flatten the features
    flattened = layers.Flatten()(fused)

    # Fully connected layer for classification
    output_layer = layers.Dense(10, activation='softmax')(flattened)  # 10 classes for MNIST

    # Create the model
    model = models.Model(inputs=input_layer, outputs=output_layer)

    return model

# Example usage
model = dl_model()
model.summary()