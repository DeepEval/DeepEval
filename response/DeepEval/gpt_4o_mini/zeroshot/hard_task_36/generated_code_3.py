import tensorflow as tf
from tensorflow.keras import layers, models

def dl_model():
    # Input layer
    input_shape = (28, 28, 1)  # MNIST images are 28x28 pixels with 1 color channel
    inputs = layers.Input(shape=input_shape)

    # Main pathway
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)  # 3x3 Conv
    x = layers.Conv2D(32, (1, 1), activation='relu', padding='same')(x)      # 1x1 Conv
    x = layers.Conv2D(32, (1, 1), activation='relu', padding='same')(x)      # Another 1x1 Conv
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)                             # Max Pooling
    x = layers.Dropout(0.5)(x)                                               # Dropout layer

    # Branch pathway
    branch = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)  # Match size with main pathway

    # Fusion of the two pathways
    combined = layers.add([x, branch])

    # Global Average Pooling and Flatten
    x = layers.GlobalAveragePooling2D()(combined)
    x = layers.Flatten()(x)

    # Fully connected layer for classification
    outputs = layers.Dense(10, activation='softmax')(x)  # 10 classes for MNIST

    # Construct the model
    model = models.Model(inputs=inputs, outputs=outputs)
    
    return model

# Example of how to create the model
model = dl_model()
model.summary()  # Display the model summary