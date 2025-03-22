# Import necessary packages
from tensorflow.keras.layers import Input, Conv2D, DepthwiseConv2D, Activation, Add, Flatten, Dense
from tensorflow.keras.models import Model

def dl_model():
    """
    This function generates a deep learning model for image classification using the Functional API of Keras.
    
    The model consists of three branches, each composed of a same block. 
    This block first elevates the dimension through a 1x1 convolutional layer, 
    then extracts features through a 3x3 depthwise separable convolutional, 
    followed by a 1x1 convolutional layer to reduce the dimension. 
    Finally, it adds the block's input to form the output.
    
    The outputs from the three branches are concatenated, 
    then passed through a flattening layer followed by a fully connected layer to generate classification probabilities.
    """
    
    # Define the input layer
    input_layer = Input(shape=(28, 28, 1))
    
    # Define the same block
    def same_block(x):
        # Elevate the dimension through a 1x1 convolutional layer
        x = Conv2D(32, (1, 1), padding='same')(x)
        
        # Extract features through a 3x3 depthwise separable convolutional
        x = DepthwiseConv2D((3, 3), padding='same')(x)
        
        # Reduce the dimension through a 1x1 convolutional layer
        x = Conv2D(32, (1, 1), padding='same')(x)
        
        # Add the block's input to form the output
        x = Add()([x, input_layer])
        
        return x
    
    # Define the three branches
    branch1 = same_block(input_layer)
    branch2 = same_block(Conv2D(32, (1, 1), padding='same')(input_layer))
    branch3 = same_block(Conv2D(32, (3, 3), padding='same')(input_layer))
    
    # Concatenate the outputs from the three branches
    concatenated = Concatenate()([branch1, branch2, branch3])
    
    # Pass the concatenated output through a flattening layer
    flattened = Flatten()(concatenated)
    
    # Pass the flattened output through a fully connected layer to generate classification probabilities
    output_layer = Dense(10, activation='softmax')(flattened)
    
    # Define the model
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Import necessary packages
from tensorflow.keras.layers import Concatenate

# Test the function
model = dl_model()
model.summary()