import tensorflow as tf
from tensorflow.keras import layers, models

def dl_model():
    # Input layer
    input_layer = layers.Input(shape=(32, 32, 3))

    # First convolutional branch (3x3)
    branch1 = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(input_layer)
    branch1 = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(branch1)

    # Second convolutional branch (5x5)
    branch2 = layers.Conv2D(32, (5, 5), padding='same', activation='relu')(input_layer)
    branch2 = layers.Conv2D(64, (5, 5), padding='same', activation='relu')(branch2)

    # Combine branches through addition
    combined = layers.Add()([branch1, branch2])

    # Global Average Pooling
    pooled = layers.GlobalAveragePooling2D()(combined)

    # Fully connected layers
    dense1 = layers.Dense(128, activation='relu')(pooled)
    dense2 = layers.Dense(10, activation='softmax')(dense1)

    # Create the model
    model = models.Model(inputs=input_layer, outputs=dense2)

    # Compile the model
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model