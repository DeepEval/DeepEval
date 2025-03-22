import tensorflow as tf
from tensorflow.keras import layers, models

def dl_model():
    # Input layer
    input_shape = (32, 32, 3)  # CIFAR-10 images are 32x32 pixels with 3 color channels (RGB)
    inputs = layers.Input(shape=input_shape)

    # Initial convolutional layer
    x = layers.Conv2D(32, (3, 3), padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # Parallel blocks
    block1 = layers.Conv2D(64, (3, 3), padding='same')(x)
    block1 = layers.BatchNormalization()(block1)
    block1 = layers.ReLU()(block1)

    block2 = layers.Conv2D(64, (3, 3), padding='same')(x)
    block2 = layers.BatchNormalization()(block2)
    block2 = layers.ReLU()(block2)

    block3 = layers.Conv2D(64, (3, 3), padding='same')(x)
    block3 = layers.BatchNormalization()(block3)
    block3 = layers.ReLU()(block3)

    # Combine outputs of parallel blocks with the initial convolution
    merged = layers.add([block1, block2, block3])
    merged = layers.add([merged, x])  # Adding initial convolution output

    # Flatten and fully connected layers
    x = layers.Flatten()(merged)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.5)(x)  # Optional dropout for regularization
    outputs = layers.Dense(10, activation='softmax')(x)  # 10 classes for CIFAR-10

    # Create the model
    model = models.Model(inputs=inputs, outputs=outputs)

    return model

# Example usage
model = dl_model()
model.summary()  # To view the model architecture