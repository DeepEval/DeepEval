import tensorflow as tf
from tensorflow.keras import layers

def dl_model():
    input_img = layers.Input(shape=(32, 32, 3))

    # Path 1: 1x1 convolution
    path1 = layers.Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(input_img)

    # Path 2: 1x1 -> 3x3 -> 3x3
    path2 = layers.Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(input_img)
    path2 = layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(path2)
    path2 = layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(path2)

    # Path 3: 1x1 -> 3x3
    path3 = layers.Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(input_img)
    path3 = layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(path3)

    # Path 4: Max pooling -> 1x1
    path4 = layers.MaxPool2D(pool_size=(3, 3), strides=2)(input_img)
    path4 = layers.Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(path4)

    # Concatenate paths
    merged = layers.concatenate([path1, path2, path3, path4], axis=-1)

    # Flatten and dense layer
    x = layers.Flatten()(merged)
    x = layers.Dense(128, activation='relu')(x)

    # Output layer
    output = layers.Dense(10, activation='softmax')(x)

    model = tf.keras.Model(inputs=input_img, outputs=output)

    return model