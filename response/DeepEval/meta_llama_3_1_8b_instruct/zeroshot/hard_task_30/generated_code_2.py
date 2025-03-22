# Import necessary packages
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    """
    Creates a deep learning model for image classification using the CIFAR-10 dataset.
    
    The model comprises two blocks. The first block features a dual-path structure: 
    the main path and a branch path. The main path starts with two convolutional layers 
    to increase the feature map width, followed by another convolutional to restore 
    the number of channels to match that of the input layer. The branch path directly 
    connects to the input. Both paths are then combined through addition to produce 
    the final output.

    The second block splits the input into three groups along the channel by 
    encapsulating tf.split within Lambda layer. Each group extracts features using 
    depthwise separable convolutional layers with different kernel sizes: 1x1, 3x3, 
    and 5x5. The outputs from these three groups are concatenated.

    After establishing the input layer, the model processes features through the two 
    blocks and concludes with two fully connected layers that generate classification 
    probabilities.

    Returns:
        model (tf.keras.Model): The constructed deep learning model.
    """

    # Define the input layer
    inputs = keras.Input(shape=(32, 32, 3))

    # Block 1: Dual-path structure
    x = layers.Conv2D(16, (3, 3), activation='relu')(inputs)  # 16 filters, kernel size 3x3
    x = layers.Conv2D(32, (3, 3), activation='relu')(x)  # 32 filters, kernel size 3x3
    main_path = layers.Conv2D(16, (3, 3), activation='relu')(x)  # Restore number of channels to 16
    branch_path = inputs

    # Combine both paths using addition
    x = layers.Add()([main_path, branch_path])

    # Block 2: Split and process features using depthwise separable convolutional layers
    x = layers.Lambda(lambda x: tf.split(x, 3, axis=-1))(x)
    x = [layers.SeparableConv2D(8, (1, 1), activation='relu')(i) for i in x]  # 8 filters, kernel size 1x1
    x = [layers.SeparableConv2D(16, (3, 3), activation='relu')(i) for i in x]  # 16 filters, kernel size 3x3
    x = [layers.SeparableConv2D(32, (5, 5), activation='relu')(i) for i in x]  # 32 filters, kernel size 5x5
    x = layers.Concatenate()(x)  # Concatenate outputs from three groups

    # Flatten the output
    x = layers.Flatten()(x)

    # Fully connected layers for classification
    x = layers.Dense(128, activation='relu')(x)  # 128 units, activation'relu'
    outputs = layers.Dense(10, activation='softmax')(x)  # 10 units, activation'softmax'

    # Define the model
    model = keras.Model(inputs=inputs, outputs=outputs)

    return model