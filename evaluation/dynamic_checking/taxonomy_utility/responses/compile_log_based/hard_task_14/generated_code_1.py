import tensorflow as tf
from tensorflow.keras import layers, models

def dl_model():
    # Input layer for CIFAR-10 images (32x32x3)
    input_layer = layers.Input(shape=(32, 32, 3))

    # Main path
    main_path = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(input_layer)
    main_path = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(main_path)
    
    # Using a Conv2D layer to maintain shape for combining
    main_path = layers.Conv2D(32, (1, 1), padding='same', activation='relu')(main_path)  # Match channels with branch_path

    # Fully connected layers in main path
    main_path = layers.GlobalAveragePooling2D()(main_path)
    main_path = layers.Dense(128, activation='relu')(main_path)
    main_path = layers.Dense(128, activation='relu')(main_path)
    main_path = layers.Dense(32, activation='sigmoid')(main_path)  # Match the number of channels with branch_path

    # Reshape to match the input layer's channels
    main_path = layers.Reshape((1, 1, 32))(main_path)  # Adjusted to 32 channels

    # Element-wise multiplication with the original feature map
    main_path = layers.Multiply()([main_path, input_layer])

    # Branch path
    branch_path = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(input_layer)

    # Combine both paths
    combined = layers.Add()([main_path, branch_path])

    # Further processing of the combined feature map
    combined = layers.GlobalAveragePooling2D()(combined)
    combined = layers.Dense(128, activation='relu')(combined)
    combined = layers.Dense(128, activation='relu')(combined)
    output_layer = layers.Dense(10, activation='softmax')(combined)  # 10 classes for CIFAR-10

    # Construct the model
    model = models.Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Example usage:
model = dl_model()
model.summary()  # To display the model architecture