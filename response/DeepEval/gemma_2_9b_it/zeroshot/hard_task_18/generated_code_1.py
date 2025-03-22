from tensorflow.keras import layers
from tensorflow.keras.models import Model

def dl_model():
    inputs = layers.Input(shape=(32, 32, 3))

    # First Block
    x = layers.Conv2D(32, (3, 3), activation='relu')(inputs)
    x = layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = layers.AveragePooling2D((2, 2))(x)

    # Second Block
    main_path = layers.GlobalAveragePooling2D()(x)
    main_path = layers.Dense(64, activation='relu')(main_path)
    main_path = layers.Dense(64, activation='relu')(main_path)
    main_path = layers.Reshape((1, 1, 64))(main_path) 

    x = layers.Multiply()([x, main_path])
    x = layers.Flatten()(x)
    outputs = layers.Dense(10, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model