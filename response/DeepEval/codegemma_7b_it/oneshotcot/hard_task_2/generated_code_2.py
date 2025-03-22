import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():

    # Input layer
    input_layer = layers.Input(shape=(32, 32, 3))

    # Split the input into three groups along the channel dimension
    split_input = layers.Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=3))(input_layer)

    # Define the parallel paths
    path1 = layers.Conv2D(filters=16, kernel_size=(1, 1), activation='relu')(split_input[0])
    path1 = layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(path1)
    path1 = layers.Conv2D(filters=16, kernel_size=(1, 1), activation='relu')(path1)

    path2 = layers.Conv2D(filters=16, kernel_size=(1, 1), activation='relu')(split_input[1])
    path2 = layers.Conv2D(filters=32, kernel_size=(5, 5), activation='relu')(path2)
    path2 = layers.Conv2D(filters=16, kernel_size=(1, 1), activation='relu')(path2)

    path3 = layers.Conv2D(filters=16, kernel_size=(1, 1), activation='relu')(split_input[2])
    path3 = layers.Conv2D(filters=32, kernel_size=(7, 7), activation='relu')(path3)
    path3 = layers.Conv2D(filters=16, kernel_size=(1, 1), activation='relu')(path3)

    # Concatenate the outputs of the parallel paths
    concat = layers.Concatenate(axis=3)([path1, path2, path3])

    # Main path
    main_path = layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(concat)
    main_path = layers.MaxPooling2D(pool_size=(2, 2))(main_path)
    main_path = layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(main_path)

    # Fusion with original input
    fusion = layers.Add()([main_path, input_layer])

    # Flatten and fully connected layer
    flatten = layers.Flatten()(fusion)
    output_layer = layers.Dense(units=10, activation='softmax')(flatten)

    # Model definition
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model