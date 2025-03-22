import tensorflow as tf
from tensorflow.keras import layers

def dl_model():
    input_shape = (32, 32, 3)  # CIFAR-10 input shape
    
    # Block 1
    inputs = layers.Input(shape=input_shape)
    x = layers.GlobalAveragePooling2D()(inputs)
    x = layers.Dense(128, activation='relu')(x)  # Adjust units as needed
    x = layers.Dense(32, activation='relu')(x) 
    x1 = layers.Reshape(input_shape)(x)  

    # Block 2
    x2 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x2)
    x2 = layers.MaxPooling2D((2, 2))(x2)

    # Fusion
    outputs = layers.Add()([x1 * inputs, x2])  # Element-wise multiplication

    # Classification layers
    outputs = layers.Flatten()(outputs)
    outputs = layers.Dense(128, activation='relu')(outputs)
    outputs = layers.Dense(10, activation='softmax')(outputs)  # 10 classes for CIFAR-10

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model