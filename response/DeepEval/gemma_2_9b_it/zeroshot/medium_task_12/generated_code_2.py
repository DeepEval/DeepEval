import tensorflow as tf
from tensorflow.keras import layers

def dl_model():
    input_img = layers.Input(shape=(32, 32, 3))  

    # Block 1
    x = layers.Conv2D(32, (3, 3), activation='relu')(input_img)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(32, (3, 3), activation='relu')(x)
    x = layers.BatchNormalization()(x)
    block1_out = layers.MaxPooling2D((2, 2))(x)

    # Block 2
    x = layers.Conv2D(64, (3, 3), activation='relu')(block1_out)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = layers.BatchNormalization()(x)
    block2_out = layers.MaxPooling2D((2, 2))(x)

    # Block 3
    x = layers.Conv2D(128, (3, 3), activation='relu')(block2_out)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(128, (3, 3), activation='relu')(x)
    x = layers.BatchNormalization()(x)
    block3_out = layers.MaxPooling2D((2, 2))(x)

    # Concatenate outputs
    x = layers.concatenate([block1_out, block2_out, block3_out], axis=-1)

    # Flatten and fully connected layers
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)
    output = layers.Dense(10, activation='softmax')(x)

    model = tf.keras.Model(inputs=input_img, outputs=output)
    return model