import tensorflow as tf
from tensorflow.keras import layers, models

def dl_model():
    # Input layer
    input_layer = layers.Input(shape=(32, 32, 3))

    # Branch 1: 1x1 convolution and 3x3 convolution
    branch1 = layers.Conv2D(32, (1, 1), activation='relu')(input_layer)
    branch1 = layers.Conv2D(64, (3, 3), activation='relu')(branch1)
    branch1 = layers.Dropout(0.5)(branch1)

    # Branch 2: 1x1 convolution, 1x7 convolution, 7x1 convolution, 3x3 convolution
    branch2 = layers.Conv2D(32, (1, 1), activation='relu')(input_layer)
    branch2 = layers.Conv2D(64, (1, 7), activation='relu')(branch2)
    branch2 = layers.Conv2D(64, (7, 1), activation='relu')(branch2)
    branch2 = layers.Conv2D(64, (3, 3), activation='relu')(branch2)
    branch2 = layers.Dropout(0.5)(branch2)

    # Branch 3: Max pooling
    branch3 = layers.MaxPooling2D(pool_size=(2, 2))(input_layer)
    branch3 = layers.Dropout(0.5)(branch3)

    # Concatenate branches
    concatenated = layers.concatenate([branch1, branch2, branch3])

    # Fully connected layers
    flatten = layers.Flatten()(concatenated)
    dense1 = layers.Dense(128, activation='relu')(flatten)
    dense2 = layers.Dense(64, activation='relu')(dense1)
    output_layer = layers.Dense(10, activation='softmax')(dense2)

    # Create model
    model = models.Model(inputs=input_layer, outputs=output_layer)

    return model

# Example of how to compile the model
model = dl_model()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])