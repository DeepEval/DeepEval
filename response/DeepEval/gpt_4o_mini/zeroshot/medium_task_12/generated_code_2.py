import tensorflow as tf
from tensorflow.keras import layers, models

def dl_model():
    input_shape = (32, 32, 3)  # CIFAR-10 image dimensions
    inputs = layers.Input(shape=input_shape)

    # Block 1
    x1 = layers.Conv2D(32, (3, 3), padding='same')(inputs)
    x1 = layers.BatchNormalization()(x1)
    x1 = layers.ReLU()(x1)
    
    # Block 2
    x2 = layers.Conv2D(32, (3, 3), padding='same')(x1)
    x2 = layers.BatchNormalization()(x2)
    x2 = layers.ReLU()(x2)
    
    # Concatenate Block 1 and Block 2
    x2 = layers.Concatenate()([x1, x2])

    # Block 3
    x3 = layers.Conv2D(64, (3, 3), padding='same')(x2)
    x3 = layers.BatchNormalization()(x3)
    x3 = layers.ReLU()(x3)
    
    # Concatenate Block 2 and Block 3
    x3 = layers.Concatenate()([x2, x3])

    # Flatten the output
    x = layers.Flatten()(x3)

    # Fully connected layers
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dense(10, activation='softmax')(x)  # 10 classes for CIFAR-10

    # Construct the model
    model = models.Model(inputs=inputs, outputs=x)

    return model

# Create the model
model = dl_model()
model.summary()  # Display the model summary