from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    input_tensor = keras.Input(shape=(28, 28, 1))

    # Pathway 1
    x1 = layers.BatchNormalization()(input_tensor)
    x1 = layers.ReLU()(x1)
    x1 = layers.Conv2D(32, (3, 3), padding='same')(x1)

    for _ in range(2):  # Repeat block twice
        x1 = layers.BatchNormalization()(x1)
        x1 = layers.ReLU()(x1)
        x1 = layers.Conv2D(32, (3, 3), padding='same')(x1)
    
    # Concatenate original input with extracted features
    x1 = layers.concatenate([input_tensor, x1], axis=-1)

    # Pathway 2
    x2 = layers.BatchNormalization()(input_tensor)
    x2 = layers.ReLU()(x2)
    x2 = layers.Conv2D(64, (3, 3), padding='same')(x2)

    for _ in range(2):
        x2 = layers.BatchNormalization()(x2)
        x2 = layers.ReLU()(x2)
        x2 = layers.Conv2D(64, (3, 3), padding='same')(x2)

    # Concatenate original input with extracted features
    x2 = layers.concatenate([input_tensor, x2], axis=-1)

    # Merge outputs from both pathways
    merged = layers.concatenate([x1, x2], axis=-1)

    # Classification layers
    merged = layers.Flatten()(merged)
    merged = layers.Dense(128, activation='relu')(merged)
    output = layers.Dense(10, activation='softmax')(merged)

    model = keras.Model(inputs=input_tensor, outputs=output)
    return model