# Import necessary packages
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    """
    This function constructs a deep learning model for image classification using the CIFAR-10 dataset.
    
    The model consists of two main components: a main path and a branch path.
    In the main path, the input is split into three groups along the last dimension.
    The first group remains unchanged, while the second group undergoes feature extraction via a 3x3 convolutional layer.
    The output of the second group is then combined with the third group before passing through an additional 3x3 convolution.
    Finally, the outputs of all three groups are concatenated to form the output of the main path.
    The branch path employs a 1x1 convolutional layer to process the input.
    The outputs from both the main and branch paths are fused together through addition.
    The final classification result is obtained by flattening the combined output and passing it through a fully connected layer.
    
    Returns:
        model (keras.Model): The constructed deep learning model.
    """

    # Define the input shape
    input_shape = (32, 32, 3)

    # Create the input layer
    inputs = keras.Input(shape=input_shape)

    # Main Path
    # Split the input into three groups along the last dimension
    split = layers.Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(inputs)
    
    # First group remains unchanged
    group1 = split[0]
    
    # Second group undergoes feature extraction via a 3x3 convolutional layer
    group2 = layers.Conv2D(64, (3, 3), activation='relu')(split[1])
    
    # Combine the output of the second group with the third group
    combined = layers.Concatenate()([split[2], group2])
    
    # Pass the combined output through an additional 3x3 convolutional layer
    combined = layers.Conv2D(64, (3, 3), activation='relu')(combined)
    
    # Concatenate the outputs of all three groups
    main_output = layers.Concatenate()([group1, combined])
    
    # Branch Path
    # Employ a 1x1 convolutional layer to process the input
    branch_output = layers.Conv2D(64, (1, 1), activation='relu')(inputs)
    
    # Fuse the outputs from both the main and branch paths through addition
    fused_output = layers.Add()([main_output, branch_output])
    
    # Flatten the combined output
    flattened_output = layers.Flatten()(fused_output)
    
    # Pass the flattened output through a fully connected layer for classification
    outputs = layers.Dense(10, activation='softmax')(flattened_output)
    
    # Create the model
    model = keras.Model(inputs=inputs, outputs=outputs)
    
    return model