import tensorflow as tf
from tensorflow.keras import layers

def dl_model():
    # Define the input shape
    input_shape = (28, 28, 1)

    # Main path
    input_layer = layers.Input(shape=input_shape)
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_layer) 
    x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(x)

    # Branch path
    branch_layer = layers.Input(shape=input_shape)
    branch_x = layers.Conv2D(16, (1, 1), activation='relu', padding='same')(branch_layer)

    # Combine paths
    merged = layers.Add()([x, branch_x])

    # Flatten and classify
    x = layers.Flatten()(merged)
    output_layer = layers.Dense(10, activation='softmax')(x)

    # Create the model
    model = tf.keras.Model(inputs=[input_layer, branch_layer], outputs=output_layer)

    return model