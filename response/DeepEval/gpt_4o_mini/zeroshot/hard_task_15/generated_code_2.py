import tensorflow as tf
from tensorflow.keras import layers, models

def dl_model():
    # Input layer
    input_layer = layers.Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 with 3 channels (RGB)

    # Main path
    main_path = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_layer)
    main_path = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(main_path)
    main_path = layers.MaxPooling2D((2, 2))(main_path)
    main_path = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(main_path)
    main_path = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(main_path)
    main_path = layers.MaxPooling2D((2, 2))(main_path)
    main_path = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(main_path)
    main_path = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(main_path)
    main_path = layers.GlobalAveragePooling2D()(main_path)

    # Fully connected layers in the main path
    main_path = layers.Dense(128, activation='relu')(main_path)
    main_path = layers.Dense(32)(main_path)

    # Reshape to the input shape to match the input layer's shape
    weights = layers.Reshape((1, 1, 32))(main_path)

    # Multiply element-wise with the input feature map
    weighted_input = layers.Multiply()([input_layer, weights])

    # Branch path
    branch_path = input_layer  # Directly connected to input layer without modification

    # Combine main path and branch path
    combined = layers.Add()([weighted_input, branch_path])

    # Final fully connected layers
    combined = layers.GlobalAveragePooling2D()(combined)  # Optional: Pooling before Dense layers
    combined = layers.Dense(128, activation='relu')(combined)
    output_layer = layers.Dense(10, activation='softmax')(combined)  # CIFAR-10 has 10 classes

    # Construct the model
    model = models.Model(inputs=input_layer, outputs=output_layer)

    return model

# Example usage:
model = dl_model()
model.summary()  # Display the model architecture