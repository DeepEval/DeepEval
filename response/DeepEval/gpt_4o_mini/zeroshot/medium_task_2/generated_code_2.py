import tensorflow as tf
from tensorflow.keras import layers, models

def dl_model():
    # Input layer
    input_layer = layers.Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 with 3 color channels

    # Main path
    main_path = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_layer)
    main_path = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(main_path)
    main_path = layers.MaxPooling2D(pool_size=(2, 2))(main_path)

    # Branch path
    branch_path = layers.Conv2D(32, (5, 5), activation='relu', padding='same')(input_layer)

    # Concatenate both paths
    combined = layers.concatenate([main_path, branch_path])

    # Flatten the combined features
    flatten = layers.Flatten()(combined)

    # Fully connected layers
    dense1 = layers.Dense(128, activation='relu')(flatten)
    dense2 = layers.Dense(10, activation='softmax')(dense1)  # 10 classes for CIFAR-10

    # Create the model
    model = models.Model(inputs=input_layer, outputs=dense2)

    return model

# Example of how to use the function to get the model
model = dl_model()
model.summary()  # Display the model architecture