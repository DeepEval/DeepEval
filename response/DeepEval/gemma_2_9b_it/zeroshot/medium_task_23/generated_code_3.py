import tensorflow as tf
from tensorflow.keras import layers

def dl_model():
    inputs = layers.Input(shape=(32, 32, 3))

    # Path 1
    path1 = layers.Conv2D(filters=16, kernel_size=(1, 1))(inputs)

    # Path 2
    path2 = layers.Conv2D(filters=16, kernel_size=(1, 1))(inputs)
    path2 = layers.Conv2D(filters=16, kernel_size=(1, 7))(path2)
    path2 = layers.Conv2D(filters=16, kernel_size=(7, 1))(path2)

    # Path 3
    path3 = layers.Conv2D(filters=16, kernel_size=(1, 1))(inputs)
    path3_1 = layers.Conv2D(filters=16, kernel_size=(1, 7))(path3)
    path3_1 = layers.Conv2D(filters=16, kernel_size=(7, 1))(path3_1)
    path3_2 = layers.Conv2D(filters=16, kernel_size=(7, 1))(path3)
    path3_2 = layers.Conv2D(filters=16, kernel_size=(1, 7))(path3_2)
    path3 = layers.concatenate([path3_1, path3_2], axis=3)

    # Path 4
    path4 = layers.AveragePooling2D(pool_size=(2, 2))(inputs)
    path4 = layers.Conv2D(filters=16, kernel_size=(1, 1))(path4)

    # Concatenate paths
    output = layers.concatenate([path1, path2, path3, path4], axis=3)

    # Flatten and classify
    output = layers.Flatten()(output)
    output = layers.Dense(units=10, activation='softmax')(output)

    model = tf.keras.Model(inputs=inputs, outputs=output)
    return model