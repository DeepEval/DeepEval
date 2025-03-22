# Import necessary packages
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    """
    This function constructs a deep learning model for image classification using Functional APIs of Keras.
    The model consists of two pathways to process features: the main path and the branch path.
    The main path includes a specific block that extracts and enhances features through a ReLU activation function and a separable convolutional layer.
    The branch path employs a convolutional layer that maintains the same channels as the main_path's output.
    The two paths fuse their features through an addition operation, and finally, the flattened features are passed through a fully connected layer to output the classification results.
    
    Returns:
    model (keras.Model): The constructed deep learning model.
    """

    # Define the input shape
    input_shape = (28, 28, 1)

    # Define the inputs
    inputs = keras.Input(shape=input_shape)

    # Main path
    x = layers.SeparableConv2D(32, 3, activation='relu', padding='same')(inputs)
    x = layers.MaxPooling2D(pool_size=2, strides=2)(x)

    for _ in range(2):
        x = layers.SeparableConv2D(64, 3, activation='relu', padding='same')(x)
        x = layers.MaxPooling2D(pool_size=2, strides=2)(x)

    # Specific block (repeat three times)
    for _ in range(3):
        x = layers.SeparableConv2D(128, 3, activation='relu', padding='same')(x)
        x = layers.Concatenate()([x, layers.SeparableConv2D(128, 3, activation='relu', padding='same')(x)])
        x = layers.MaxPooling2D(pool_size=2, strides=2)(x)

    # Branch path
    branch_x = layers.Conv2D(128, 3, padding='same')(x)
    branch_x = layers.GlobalAveragePooling2D()(branch_x)

    # Fuse features from main path and branch path
    x = layers.Concatenate()([x, branch_x])

    # Flatten and dense layer
    x = layers.Flatten()(x)
    outputs = layers.Dense(10, activation='softmax')(x)

    # Define the model
    model = keras.Model(inputs=inputs, outputs=outputs)

    return model