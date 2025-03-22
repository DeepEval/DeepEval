import tensorflow as tf
from tensorflow.keras import layers, models

def dl_model():
    # Input layer
    input_layer = layers.Input(shape=(32, 32, 3))

    # First Convolutional layer
    x = layers.Conv2D(32, (3, 3), activation='relu')(input_layer)

    # Max Pooling 1x1
    pooling_1 = layers.MaxPooling2D(pool_size=(1, 1), strides=(1, 1))(x)
    flat_1 = layers.Flatten()(pooling_1)

    # Max Pooling 2x2
    pooling_2 = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    flat_2 = layers.Flatten()(pooling_2)

    # Max Pooling 4x4
    pooling_3 = layers.MaxPooling2D(pool_size=(4, 4), strides=(4, 4))(x)
    flat_3 = layers.Flatten()(pooling_3)

    # Concatenate the flattened outputs
    concatenated = layers.concatenate([flat_1, flat_2, flat_3])

    # Fully connected layer 1
    fc_1 = layers.Dense(128, activation='relu')(concatenated)

    # Fully connected layer 2
    fc_2 = layers.Dense(64, activation='relu')(fc_1)

    # Output layer for classification
    output_layer = layers.Dense(10, activation='softmax')(fc_2)

    # Create the model
    model = models.Model(inputs=input_layer, outputs=output_layer)

    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model

# Example of how to create the model
model = dl_model()
model.summary()