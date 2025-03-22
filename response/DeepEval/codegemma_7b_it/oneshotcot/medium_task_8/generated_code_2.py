import keras
from keras import layers

def dl_model():
    input_layer = keras.Input(shape=(32, 32, 3))

    # Main path
    path_main = input_layer
    path_main = layers.Lambda(lambda x: tf.split(x, 3, axis=-1))(path_main)
    path_main = layers.Lambda(lambda x: tf.concat([x[0], 
                                    keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same')(x[1]),
                                    keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same')(x[2])], axis=-1))(path_main)
    path_main = keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same')(path_main)

    # Branch path
    path_branch = keras.layers.Conv2D(filters=64, kernel_size=(1, 1), padding='same')(input_layer)

    # Fusion
    concat_path = layers.add([path_main, path_branch])

    # Classification
    concat_path = keras.layers.Flatten()(concat_path)
    output_layer = keras.layers.Dense(units=10, activation='softmax')(concat_path)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model