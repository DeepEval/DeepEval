import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    input_img = keras.Input(shape=(28, 28, 1))  

    # Main Pathway
    x = layers.Conv2D(32, (3, 3), activation='relu')(input_img)
    x = layers.Conv2D(64, (1, 1), activation='relu')(x)
    x = layers.Conv2D(64, (1, 1), activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.5)(x)

    # Branch Pathway
    branch_x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(input_img)

    # Feature Fusion
    x = layers.Add()([x, branch_x])

    # Global Average Pooling
    x = layers.GlobalAveragePooling2D()(x)

    # Flatten
    x = layers.Flatten()(x)

    # Output Layer
    output = layers.Dense(10, activation='softmax')(x)

    model = keras.Model(inputs=input_img, outputs=output)
    return model