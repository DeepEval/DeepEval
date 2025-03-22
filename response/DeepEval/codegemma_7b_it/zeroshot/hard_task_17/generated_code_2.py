import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    inputs = keras.Input(shape=(32, 32, 3))

    # Block 1
    x = layers.Conv2D(filters=64, kernel_size=3, padding='same', activation='relu')(inputs)
    x = layers.Conv2D(filters=64, kernel_size=3, padding='same', activation='relu')(x)
    pooled_features = layers.GlobalAveragePooling2D()(x)
    fc_layer_1 = layers.Dense(units=64, activation='relu')(pooled_features)
    fc_layer_2 = layers.Dense(units=64, activation='relu')(fc_layer_1)
    weights = layers.Reshape((1, 1, 64))(fc_layer_2)
    weighted_features = layers.Multiply()([weights, x])

    # Block 2
    x = layers.Conv2D(filters=128, kernel_size=3, padding='same', activation='relu')(weighted_features)
    x = layers.Conv2D(filters=128, kernel_size=3, padding='same', activation='relu')(x)
    x = layers.MaxPooling2D()(x)

    # Branch from Block 1
    branch_features = layers.Conv2D(filters=128, kernel_size=3, padding='same', activation='relu')(pooled_features)
    branch_features = layers.Conv2D(filters=128, kernel_size=3, padding='same', activation='relu')(branch_features)
    branch_features = layers.MaxPooling2D()(branch_features)

    # Fusion
    combined_features = layers.Add()([x, branch_features])

    # Output layer
    outputs = layers.Dense(units=10, activation='softmax')(combined_features)

    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model