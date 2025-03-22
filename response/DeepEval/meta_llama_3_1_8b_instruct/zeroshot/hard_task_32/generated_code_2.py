# Import necessary packages
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf

def dl_model():
    """
    This function constructs a deep learning model with three branches for image classification.
    Each branch is built from a specialized block that includes a depthwise separable convolutional layer
    followed by a 1x1 convolutional layer and dropout layers to mitigate overfitting.
    The outputs from the three branches are concatenated and then processed through two fully connected layers
    to generate the final classification results.
    
    Returns:
        model (keras.Model): The constructed deep learning model.
    """

    # Define the input shape for the MNIST dataset
    input_shape = (28, 28, 1)

    # Create an input layer for the model
    inputs = keras.Input(shape=input_shape, name='input_layer')

    # Define the first branch
    x1 = layers.SeparableConv2D(32, (3, 3), activation='relu', name='branch1_conv1')(inputs)
    x1 = layers.Dropout(0.2, name='branch1_dropout1')(x1)
    x1 = layers.Conv2D(32, (1, 1), activation='relu', name='branch1_conv2')(x1)
    x1 = layers.Dropout(0.2, name='branch1_dropout2')(x1)

    # Define the second branch
    x2 = layers.SeparableConv2D(32, (5, 5), activation='relu', name='branch2_conv1')(inputs)
    x2 = layers.Dropout(0.2, name='branch2_dropout1')(x2)
    x2 = layers.Conv2D(32, (1, 1), activation='relu', name='branch2_conv2')(x2)
    x2 = layers.Dropout(0.2, name='branch2_dropout2')(x2)

    # Define the third branch
    x3 = layers.SeparableConv2D(32, (7, 7), activation='relu', name='branch3_conv1')(inputs)
    x3 = layers.Dropout(0.2, name='branch3_dropout1')(x3)
    x3 = layers.Conv2D(32, (1, 1), activation='relu', name='branch3_conv2')(x3)
    x3 = layers.Dropout(0.2, name='branch3_dropout2')(x3)

    # Concatenate the outputs from the three branches
    x = layers.Concatenate(name='concat_branches')([x1, x2, x3])

    # Flatten the concatenated output
    x = layers.Flatten(name='flatten')(x)

    # Define two fully connected layers for classification
    x = layers.Dense(128, activation='relu', name='fc1')(x)
    x = layers.Dense(10, activation='softmax', name='fc2')(x)

    # Create the model by adding the input and output layers
    model = keras.Model(inputs=inputs, outputs=x, name='deep_learning_model')

    return model

# Create and print the constructed model
model = dl_model()
model.summary()