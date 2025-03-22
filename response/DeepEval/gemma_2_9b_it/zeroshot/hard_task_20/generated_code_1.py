import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    input_tensor = keras.Input(shape=(32, 32, 3))

    # Main path
    x = layers.Lambda(lambda x: tf.split(x, 3, axis=1))(input_tensor)
    
    # Group 1 (1x1 conv)
    branch1 = layers.Conv2D(32, kernel_size=(1, 1), activation='relu')(x[0])
    
    # Group 2 (3x3 conv)
    branch2 = layers.Conv2D(32, kernel_size=(3, 3), activation='relu')(x[1])

    # Group 3 (5x5 conv)
    branch3 = layers.Conv2D(32, kernel_size=(5, 5), activation='relu')(x[2])
    
    x = layers.concatenate([branch1, branch2, branch3], axis=1)

    # Branch path
    branch_output = layers.Conv2D(32, kernel_size=(1, 1), activation='relu')(input_tensor)

    # Combine outputs
    x = layers.Add()([x, branch_output])

    # Fully connected layers
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)
    output = layers.Dense(10, activation='softmax')(x)

    model = keras.Model(inputs=input_tensor, outputs=output)
    return model

# Get the model
model = dl_model()

# Print the model summary
model.summary()