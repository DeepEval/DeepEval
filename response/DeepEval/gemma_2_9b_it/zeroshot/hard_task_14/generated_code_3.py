import tensorflow as tf
from tensorflow.keras import layers, models

def dl_model():
    inputs = layers.Input(shape=(32, 32, 3))

    # Main Path
    x = layers.Conv2D(32, kernel_size=(3, 3), activation='relu')(inputs)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Conv2D(64, kernel_size=(3, 3), activation='relu')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(32, activation='relu')(x)
    x = layers.Dense(32, activation='relu')(x)
    main_output = layers.Reshape((32, 32, 32))(x)

    # Branch Path
    branch_output = layers.Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same')(inputs)

    # Add Outputs
    output = layers.Add()([main_output, branch_output])

    # Classification Layers
    output = layers.Flatten()(output)
    output = layers.Dense(128, activation='relu')(output)
    output = layers.Dense(10, activation='softmax')(output)

    model = models.Model(inputs=inputs, outputs=output)

    return model