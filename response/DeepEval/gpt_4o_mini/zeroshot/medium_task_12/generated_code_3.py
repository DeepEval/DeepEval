import tensorflow as tf
from tensorflow.keras import layers, models

def dl_model():
    # Input layer
    input_img = layers.Input(shape=(32, 32, 3))

    # Block 1
    x1 = layers.Conv2D(32, (3, 3), padding='same')(input_img)
    x1 = layers.BatchNormalization()(x1)
    x1 = layers.ReLU()(x1)

    # Block 2
    x2 = layers.Conv2D(64, (3, 3), padding='same')(x1)
    x2 = layers.BatchNormalization()(x2)
    x2 = layers.ReLU()(x2)

    # Concatenate block 1 and block 2
    x2_concat = layers.Concatenate()([x1, x2])

    # Block 3
    x3 = layers.Conv2D(128, (3, 3), padding='same')(x2_concat)
    x3 = layers.BatchNormalization()(x3)
    x3 = layers.ReLU()(x3)

    # Concatenate block 2 and block 3
    x3_concat = layers.Concatenate()([x2_concat, x3])

    # Flatten the output
    x_flat = layers.Flatten()(x3_concat)

    # Fully connected layer 1
    x_fc1 = layers.Dense(256, activation='relu')(x_flat)

    # Fully connected layer 2
    x_fc2 = layers.Dense(10, activation='softmax')(x_fc1)

    # Create the model
    model = models.Model(inputs=input_img, outputs=x_fc2)

    return model

# Create the model
model = dl_model()
model.summary()