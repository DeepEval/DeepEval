import tensorflow as tf
from tensorflow.keras import layers

def dl_model():
    inputs = layers.Input(shape=(32, 32, 3))  

    # Main Path
    x = layers.Conv2D(32, (3, 3), activation='relu')(inputs)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.GlobalAveragePooling2D()(x)  
    x = layers.Dense(32, activation='relu')(x)
    x = layers.Dense(3 * 32)(x) 
    x = tf.reshape(x, (-1, 32, 32, 3))  

    # Branch Path
    branch = layers.Conv2D(32, (3, 3), activation='relu')(inputs)
    branch = layers.Conv2D(32, (3, 3), activation='relu')(branch)

    # Add Outputs
    x = layers.add([x, branch])

    # Final Classification Layers
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)
    outputs = layers.Dense(10, activation='softmax')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model