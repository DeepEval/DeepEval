import tensorflow as tf
from tensorflow.keras import layers

def dl_model():
    # Define the basic block
    def basic_block(x):
        x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        return x

    # Input layer
    inputs = layers.Input(shape=(32, 32, 3))

    # Initial convolution to reduce dimensionality
    x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(inputs)

    # Main path
    main_path = basic_block(x)
    main_path = basic_block(main_path)

    # Branch
    branch = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    branch = basic_block(branch)

    # Feature fusion
    x = layers.Add()([main_path, branch])

    # Average pooling
    x = layers.AveragePooling2D((8, 8))(x)

    # Flatten and fully connected layer
    x = layers.Flatten()(x)
    outputs = layers.Dense(10, activation='softmax')(x)

    # Create the model
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

# Example usage
model = dl_model()
model.summary()