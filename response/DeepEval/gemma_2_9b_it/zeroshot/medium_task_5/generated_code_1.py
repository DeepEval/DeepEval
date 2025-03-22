from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    input_tensor = keras.Input(shape=(32, 32, 3))

    # Main path
    x = layers.Conv2D(32, (3, 3), activation='relu')(input_tensor)
    x = layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(128, (3, 3), activation='relu')(x)
    x = layers.Conv2D(128, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # Branch path
    branch_x = layers.Conv2D(64, (3, 3), activation='relu')(input_tensor)
    branch_x = layers.MaxPooling2D((2, 2))(branch_x)

    # Combine paths
    x = layers.Add()([x, branch_x])

    # Flatten and fully connected layers
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)
    output_tensor = layers.Dense(10, activation='softmax')(x)

    model = keras.Model(inputs=input_tensor, outputs=output_tensor)
    
    return model