import tensorflow as tf
from tensorflow.keras import layers

def dl_model():
    inputs = layers.Input(shape=(32, 32, 3))  # Assuming CIFAR-10 input shape

    # Reshape the input tensor
    x = layers.Reshape((32, 32, 3, 1))(inputs)  

    # Permute dimensions
    x = layers.Permute((2, 3, 1, 4))(x) 

    # Reshape back
    x = layers.Reshape((32, 32, 3))(x) 

    # Fully connected layer with softmax
    x = layers.Flatten()(x)
    outputs = layers.Dense(10, activation='softmax')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model