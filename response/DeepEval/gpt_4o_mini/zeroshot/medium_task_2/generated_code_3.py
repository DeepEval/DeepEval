import tensorflow as tf
from tensorflow.keras import layers, models

def dl_model():
    # Input layer
    input_layer = layers.Input(shape=(32, 32, 3))

    # Main path with two 3x3 convolutional layers and a max pooling layer
    main_path = layers.Conv2D(32, (3, 3), activation='relu')(input_layer)
    main_path = layers.Conv2D(32, (3, 3), activation='relu')(main_path)
    main_path = layers.MaxPooling2D(pool_size=(2, 2))(main_path)

    # Branch path with a single 5x5 convolutional layer
    branch_path = layers.Conv2D(32, (5, 5), activation='relu')(input_layer)

    # Combine both paths
    combined = layers.concatenate([main_path, branch_path])

    # Flatten the combined output
    flattened = layers.Flatten()(combined)

    # Fully connected layers
    dense1 = layers.Dense(128, activation='relu')(flattened)
    dense2 = layers.Dense(10, activation='softmax')(dense1)

    # Construct the model
    model = models.Model(inputs=input_layer, outputs=dense2)

    return model

# Example of how to use the model
model = dl_model()
model.summary()  # This will print the model architecture