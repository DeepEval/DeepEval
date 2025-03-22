from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    input_tensor = keras.Input(shape=(32, 32, 3))

    # Block 1
    x = layers.GlobalAveragePooling2D()(input_tensor)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dense(32, activation='relu')(x)
    x = layers.Reshape((32, 32, 3))(x)
    weighted_features = x * input_tensor 

    # Block 2
    y = layers.Conv2D(64, (3, 3), activation='relu')(input_tensor)
    y = layers.Conv2D(32, (3, 3), activation='relu')(y)
    y = layers.MaxPooling2D((2, 2))(y)

    # Fusion
    combined_features = layers.Add()([weighted_features, y])

    # Output Layers
    z = layers.Flatten()(combined_features)
    z = layers.Dense(128, activation='relu')(z)
    output_tensor = layers.Dense(10, activation='softmax')(z)

    model = keras.Model(inputs=input_tensor, outputs=output_tensor)
    return model