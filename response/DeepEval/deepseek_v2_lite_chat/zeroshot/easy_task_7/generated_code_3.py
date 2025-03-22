import tensorflow as tf
from tensorflow.keras import layers, models


def dl_model():
    # Input layer
    inputs = tf.keras.Input(shape=(28, 28, 1))

    # Main path
    x = layers.Conv2D(32, kernel_size=(3, 3), activation='relu')(inputs)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Conv2D(64, kernel_size=(3, 3), activation='relu')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Conv2D(64, kernel_size=(3, 3), activation='relu')(x)
    x = layers.Dropout(0.2)(x)  # Add dropout for regularization

    # Branch path
    branch_output = layers.Conv2D(64, kernel_size=(3, 3), activation='relu')(inputs)
    branch_output = layers.MaxPooling2D(pool_size=(2, 2))(branch_output)
    branch_output = layers.Conv2D(64, kernel_size=(3, 3), activation='relu')(branch_output)
    branch_output = layers.MaxPooling2D(pool_size=(2, 2))(branch_output)

    # Add paths
    x = layers.add([x, branch_output])  # Add branch output to main path output

    # Flatten and fully connected layers
    x = layers.Flatten()(x)
    x = layers.Dense(64, activation='relu')(x)
    outputs = layers.Dense(10, activation='softmax')(x)  # 10 classes for MNIST

    # Model
    model = models.Model(inputs=inputs, outputs=outputs)
    return model

# Instantiate and return the model
model = dl_model()
model.summary()