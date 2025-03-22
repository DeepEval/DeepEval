from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    inputs = keras.Input(shape=(32, 32, 3))

    # First Block: Parallel Feature Extraction
    x_1x1 = layers.Conv2D(filters=32, kernel_size=1, activation='relu')(inputs)
    x_3x3 = layers.Conv2D(filters=32, kernel_size=3, activation='relu')(inputs)
    x_5x5 = layers.Conv2D(filters=32, kernel_size=5, activation='relu')(inputs)
    x_pool = layers.MaxPooling2D(pool_size=(3, 3))(inputs)

    # Concatenate outputs from parallel branches
    x = layers.Concatenate()([x_1x1, x_3x3, x_5x5, x_pool])

    # Second Block: Global Average Pooling and Dense Layers
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(units=128, activation='relu')(x)
    x = layers.Dense(units=128, activation='relu')(x)
    # Reshape and multiply with input feature map
    x = layers.Reshape((32, 32, 128))(x) 
    x = layers.multiply([x, inputs])
    # Final classification layer
    outputs = layers.Dense(units=10, activation='softmax')(x)

    model = keras.Model(inputs=inputs, outputs=outputs)

    return model