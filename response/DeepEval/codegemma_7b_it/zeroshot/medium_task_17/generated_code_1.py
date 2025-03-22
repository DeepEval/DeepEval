from tensorflow import keras
from tensorflow.keras import layers

def dl_model():

    # Define the input layer
    inputs = keras.Input(shape=(None, None, 3))

    # Reshape the input tensor into three groups
    x = layers.Reshape((-1, 3, 3))(inputs)

    # Swap the third and fourth dimensions using a permutation operation
    x = layers.Permute((0, 1, 3, 2))(x)

    # Reshape the tensor back to its original input shape
    x = layers.Reshape((None, None, 3))(x)

    # Apply a fully connected layer with softmax activation
    outputs = layers.Dense(10, activation='softmax')(x)

    # Create the model
    model = keras.Model(inputs=inputs, outputs=outputs)

    return model

# Create the model
model = dl_model()