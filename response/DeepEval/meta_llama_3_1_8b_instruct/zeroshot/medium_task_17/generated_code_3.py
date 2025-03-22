# Import necessary packages
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    # Define input layer with shape (32, 32, 3) for CIFAR-10 dataset
    inputs = keras.Input(shape=(32, 32, 3))

    # Reshape the input tensor into three groups
    reshaped = layers.Reshape((8, 8, 3, 2)) (inputs)

    # Swap the third and fourth dimensions to enable channel shuffling
    permuted = layers.Permute((3, 1, 2, 4)) (reshaped)

    # Reshape the tensor back to its original input shape
    reshaped_back = layers.Reshape((8, 8, 6)) (permuted)

    # Pass the reshaped tensor through a fully connected layer with softmax activation
    outputs = layers.Dense(10, activation='softmax') (layers.Flatten()(reshaped_back))

    # Define the model
    model = keras.Model(inputs=inputs, outputs=outputs)

    return model

# Test the model
model = dl_model()
model.summary()