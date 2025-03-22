import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    inputs = keras.Input(shape=(28, 28, 1))  

    # Main Pathway
    x = layers.Conv2D(32, (3, 3), activation='relu')(inputs)
    x = layers.Conv2D(64, (1, 1), activation='relu')(x)
    x = layers.Conv2D(64, (1, 1), activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.5)(x)

    # Branch Pathway
    branch_x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)

    # Feature Fusion
    x = layers.Add()([x, branch_x])

    # Global Average Pooling
    x = layers.GlobalAveragePooling2D()(x)

    # Flatten and Output Layer
    x = layers.Flatten()(x)
    outputs = layers.Dense(10, activation='softmax')(x)

    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

# Get the model
model = dl_model()