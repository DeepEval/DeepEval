# Import necessary packages
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf

def dl_model():
    """
    This function creates a deep learning model for image classification using the CIFAR-10 dataset.
    
    The model consists of two main components: a main path and a branch path.
    In the main path, the input is split into three groups along the last dimension.
    The first group remains unchanged, while the second group undergoes feature extraction via a 3x3 convolutional layer.
    The output of the second group is then combined with the third group before passing through an additional 3x3 convolution.
    Finally, the outputs of all three groups are concatenated to form the output of the main path.
    The branch path employs a 1x1 convolutional layer to process the input.
    The outputs from both the main and branch paths are fused together through addition.
    The final classification result is obtained by flattening the combined output and passing it through a fully connected layer.
    
    Returns:
        model (tf.keras.Model): The constructed deep learning model.
    """
    
    # Define the input shape
    input_shape = (32, 32, 3)
    
    # Define the main path
    main_input = keras.Input(shape=input_shape)
    
    # Split the input into three groups along the last dimension
    split_output = layers.Lambda(lambda x: tf.split(x, 3, axis=-1))(main_input)
    
    # Apply a 3x3 convolutional layer to the second group
    conv2d_1 = layers.Conv2D(32, (3, 3), activation='relu')(split_output[1])
    
    # Combine the output of the second group with the third group
    combined_output = layers.Concatenate()([split_output[0], split_output[2], conv2d_1])
    
    # Apply an additional 3x3 convolutional layer
    conv2d_2 = layers.Conv2D(32, (3, 3), activation='relu')(combined_output)
    
    # Concatenate the outputs of all three groups
    concatenated_output = layers.Concatenate()([split_output[0], split_output[1], split_output[2], conv2d_2])
    
    # Define the branch path
    branch_input = keras.Input(shape=input_shape)
    branch_output = layers.Conv2D(32, (1, 1), activation='relu')(branch_input)
    
    # Fuse the outputs from both the main and branch paths
    fused_output = layers.Add()([concatenated_output, branch_output])
    
    # Flatten the combined output
    flattened_output = layers.Flatten()(fused_output)
    
    # Apply a fully connected layer for classification
    classification_output = layers.Dense(10, activation='softmax')(flattened_output)
    
    # Define the model
    model = keras.Model(inputs=[main_input, branch_input], outputs=classification_output)
    
    return model