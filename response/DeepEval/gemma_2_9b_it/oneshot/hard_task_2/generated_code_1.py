import tensorflow as tf
from tensorflow.keras import layers

def dl_model():
    input_layer = layers.Input(shape=(32, 32, 3))

    # Split the input into three groups
    split_tensor = layers.Lambda(lambda x: tf.split(x, 3, axis=2))(input_layer)

    # Process each group of channels
    group1 = layers.Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(split_tensor[0])
    group1 = layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(group1)
    group1 = layers.Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(group1)

    group2 = layers.Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(split_tensor[1])
    group2 = layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(group2)
    group2 = layers.Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(group2)

    group3 = layers.Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(split_tensor[2])
    group3 = layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(group3)
    group3 = layers.Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(group3)

    # Combine the outputs
    combined = layers.Add()([group1, group2, group3])

    # Fuse with the original input
    main_path = layers.Add()([combined, input_layer])

    # Flatten and classify
    flatten_layer = layers.Flatten()(main_path)
    output_layer = layers.Dense(units=10, activation='softmax')(flatten_layer)

    model = tf.keras.Model(inputs=input_layer, outputs=output_layer)

    return model