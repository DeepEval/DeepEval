import tensorflow as tf
from tensorflow.keras import layers

def dl_model():
    input_tensor = layers.Input(shape=(32, 32, 3))

    # Path 1
    path1 = layers.Conv2D(filters=16, kernel_size=(1, 1))(input_tensor)

    # Path 2
    path2 = layers.Conv2D(filters=16, kernel_size=(1, 1))(input_tensor)
    path2 = layers.Conv2D(filters=16, kernel_size=(1, 7))(path2)
    path2 = layers.Conv2D(filters=16, kernel_size=(7, 1))(path2)

    # Path 3
    path3 = layers.Conv2D(filters=16, kernel_size=(1, 1))(input_tensor)
    path3_1 = layers.Conv2D(filters=16, kernel_size=(1, 7))(path3)
    path3_1 = layers.Conv2D(filters=16, kernel_size=(7, 1))(path3_1)
    path3_2 = layers.Conv2D(filters=16, kernel_size=(7, 1))(path3)
    path3_2 = layers.Conv2D(filters=16, kernel_size=(1, 7))(path3_2)
    path3 = layers.add([path3_1, path3_2])

    # Path 4
    path4 = layers.AveragePooling2D(pool_size=(2, 2))(input_tensor)
    path4 = layers.Conv2D(filters=16, kernel_size=(1, 1))(path4)

    # Concatenate all paths
    output = layers.concatenate([path1, path2, path3, path4])

    # Flatten and classify
    output = layers.Flatten()(output)
    output = layers.Dense(units=10, activation='softmax')(output)

    model = tf.keras.Model(inputs=input_tensor, outputs=output)
    return model