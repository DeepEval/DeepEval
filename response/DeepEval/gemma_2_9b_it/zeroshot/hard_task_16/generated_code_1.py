import tensorflow as tf
from tensorflow.keras import layers

def dl_model():
    input_tensor = layers.Input(shape=(32, 32, 3))

    # Block 1
    x = layers.Lambda(lambda x: tf.split(x, 3, axis=3))(input_tensor) 
    x = [
        layers.Sequential([
            layers.Conv2D(64, (1, 1), activation='relu'),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.Conv2D(64, (1, 1), activation='relu')
        ])(i) for i in x
    ]
    x = layers.Concatenate(axis=3)(x)

    # Transition Convolution
    x = layers.Conv2D(64, (1, 1), activation='relu')(x)

    # Block 2
    x = layers.GlobalMaxPooling2D()(x)
    x = layers.Dense(64, activation='relu')(x) 
    x = layers.Dense(64, activation='relu')(x) 

    # Generate channel-matching weights
    weights = layers.Reshape((x.shape[1], x.shape[1]))(x)

    # Multiply with adjusted output
    x = tf.multiply(x, weights)

    # Branch connection
    branch_output = layers.Conv2D(10, (1, 1), activation='relu')(input_tensor)

    # Add outputs
    x = layers.Add()([x, branch_output])

    # Output layer
    output = layers.Dense(10, activation='softmax')(x)

    model = tf.keras.Model(inputs=input_tensor, outputs=output)

    return model

# Get the model
model = dl_model()