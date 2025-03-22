import tensorflow as tf
from tensorflow.keras import layers

def dl_model():
    input_tensor = layers.Input(shape=(32, 32, 3))

    # Split the input into three groups along the channel dimension
    split_tensor = layers.Lambda(lambda x: tf.split(x, 3, axis=2))(input_tensor)

    # Process each group
    group1 = layers.Conv2D(filters=16, kernel_size=(1, 1), activation='relu')(split_tensor[0])
    group1 = layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(group1)
    group1 = layers.Dropout(0.2)(group1)

    group2 = layers.Conv2D(filters=16, kernel_size=(1, 1), activation='relu')(split_tensor[1])
    group2 = layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(group2)
    group2 = layers.Dropout(0.2)(group2)

    group3 = layers.Conv2D(filters=16, kernel_size=(1, 1), activation='relu')(split_tensor[2])
    group3 = layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(group3)
    group3 = layers.Dropout(0.2)(group3)

    # Concatenate the outputs from the three groups
    concatenated = layers.Concatenate(axis=2)([group1, group2, group3])

    # Branch pathway
    branch = layers.Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(input_tensor)

    # Combine the main pathway and branch pathway
    combined = layers.Add()([concatenated, branch])

    # Flatten and classify
    output = layers.Flatten()(combined)
    output = layers.Dense(units=10, activation='softmax')(output)

    model = tf.keras.Model(inputs=input_tensor, outputs=output)

    return model