import tensorflow as tf
from tensorflow.keras import layers, models

def dl_model():
    # Input layer
    input_layer = layers.Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 pixels with 3 color channels
    
    # Main path: Two consecutive 3x3 convolutional layers + max pooling
    main_path = layers.Conv2D(32, (3, 3), activation='relu')(input_layer)
    main_path = layers.Conv2D(32, (3, 3), activation='relu')(main_path)
    main_path = layers.MaxPooling2D(pool_size=(2, 2))(main_path)

    # Branch path: Single 5x5 convolutional layer
    branch_path = layers.Conv2D(32, (5, 5), activation='relu')(input_layer)

    # Merge the two paths
    merged = layers.Concatenate()([main_path, branch_path])
    
    # Flatten the merged output
    flattened = layers.Flatten()(merged)
    
    # Fully connected layers
    dense_1 = layers.Dense(128, activation='relu')(flattened)
    output_layer = layers.Dense(10, activation='softmax')(dense_1)  # 10 classes for CIFAR-10

    # Create the model
    model = models.Model(inputs=input_layer, outputs=output_layer)

    # Compile the model
    model.compile(optimizer='adam', 
                  loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy'])

    return model

# Example usage:
model = dl_model()
model.summary()  # Print the model summary to verify the architecture