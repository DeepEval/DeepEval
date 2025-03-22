import tensorflow as tf
from tensorflow.keras import layers, models

def basic_block(x):
    # Main path
    main_path = layers.Conv2D(16, (3, 3), padding='same')(x)
    main_path = layers.BatchNormalization()(main_path)
    main_path = layers.ReLU()(main_path)
    
    # Branch (identity connection)
    branch = x
    
    # Combine paths
    x = layers.Add()([main_path, branch])
    return layers.ReLU()(x)

def dl_model():
    # Input layer
    inputs = layers.Input(shape=(32, 32, 3))

    # First level: Single basic block
    x = layers.Conv2D(16, (3, 3), padding='same')(inputs)
    x = basic_block(x)

    # Second level: Two residual blocks
    for _ in range(2):
        main_path = basic_block(x)
        # Branch: independent convolutional layer
        branch = layers.Conv2D(16, (1, 1), padding='same')(x)
        x = layers.Add()([main_path, branch])
    
    # Third level: Global branch with convolutional layer
    global_branch = layers.Conv2D(16, (3, 3), padding='same')(inputs)
    x = layers.Add()([x, global_branch])
    
    # Average pooling and fully connected layer
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(10, activation='softmax')(x)

    # Constructing the model
    model = models.Model(inputs, outputs)

    return model

# Example usage:
model = dl_model()
model.summary()