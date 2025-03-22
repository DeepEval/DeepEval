# Import necessary packages
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    # Define the input shape of the CIFAR-10 dataset
    input_shape = (32, 32, 3)
    
    # Split the input into three groups along the last dimension
    inputs = keras.Input(shape=input_shape)
    x = layers.Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(inputs)
    
    # Feature extraction via depthwise separable convolutional layers
    group1 = x[0]
    group1 = layers.DepthwiseConv2D(kernel_size=1, padding='same')(group1)
    group1 = layers.Conv2D(64, kernel_size=1, padding='same')(group1)
    
    group2 = x[1]
    group2 = layers.DepthwiseConv2D(kernel_size=3, padding='same')(group2)
    group2 = layers.Conv2D(64, kernel_size=1, padding='same')(group2)
    
    group3 = x[2]
    group3 = layers.DepthwiseConv2D(kernel_size=5, padding='same')(group3)
    group3 = layers.Conv2D(64, kernel_size=1, padding='same')(group3)
    
    # Concatenate and fuse the features from the three groups
    x = layers.Concatenate()([group1, group2, group3])
    
    # Flatten the features into a one-dimensional vector
    x = layers.Flatten()(x)
    
    # Classification via a fully connected layer
    outputs = layers.Dense(10, activation='softmax')(x)
    
    # Define the model
    model = keras.Model(inputs=inputs, outputs=outputs)
    
    return model

# Test the model
if __name__ == "__main__":
    model = dl_model()
    print(model.summary())