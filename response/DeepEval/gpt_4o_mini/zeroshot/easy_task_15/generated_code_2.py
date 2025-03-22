import tensorflow as tf
from tensorflow.keras import layers, models

def dl_model():
    # Input layer
    inputs = layers.Input(shape=(28, 28, 1))  # MNIST images are 28x28 pixels and 1 channel (grayscale)

    # First specialized block
    x = layers.Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu')(inputs)
    x = layers.Conv2D(64, kernel_size=(1, 1), padding='same', activation='relu')(x)
    x = layers.Conv2D(64, kernel_size=(1, 1), padding='same', activation='relu')(x)
    x = layers.AveragePooling2D(pool_size=(2, 2))(x)
    x = layers.Dropout(0.25)(x)

    # Second specialized block
    x = layers.Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu')(x)
    x = layers.Conv2D(128, kernel_size=(1, 1), padding='same', activation='relu')(x)
    x = layers.Conv2D(128, kernel_size=(1, 1), padding='same', activation='relu')(x)
    x = layers.AveragePooling2D(pool_size=(2, 2))(x)
    x = layers.Dropout(0.25)(x)

    # Global Average Pooling layer
    x = layers.GlobalAveragePooling2D()(x)

    # Fully connected layers
    x = layers.Flatten()(x)
    outputs = layers.Dense(10, activation='softmax')(x)  # 10 classes for digits 0-9

    # Construct the model
    model = models.Model(inputs=inputs, outputs=outputs)

    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model