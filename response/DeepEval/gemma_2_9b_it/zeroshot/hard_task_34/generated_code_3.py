import tensorflow as tf
from tensorflow.keras import layers

def dl_model():
    input_img = layers.Input(shape=(28, 28, 1)) 

    # Main Path
    x = layers.Conv2D(32, (3, 3), activation='relu')(input_img)
    for _ in range(3):
        x = layers.SeparableConv2D(32, (3, 3), activation='relu')(x)
        x = layers.concatenate([x, layers.Conv2D(32, (3, 3), activation='relu')(input_img)], axis=3)

    # Branch Path
    branch = layers.Conv2D(32, (3, 3), activation='relu')(input_img)

    # Fusion
    x = layers.add([x, branch])

    x = layers.Flatten()(x)
    output = layers.Dense(10, activation='softmax')(x)

    model = tf.keras.Model(inputs=input_img, outputs=output)
    return model