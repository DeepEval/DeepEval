# Import necessary packages
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate, Add, Flatten, Dense
from tensorflow.keras.models import Model

def dl_model():
    """
    This function constructs a deep learning model for image classification using Functional APIs of Keras.
    
    The model structure includes a main path and a branch path. The main path first processes the input through a 1x1 convolutional layer, 
    then splits into three branches. The first branch extracts local features through a 3x3 convolutional layer, while the second and third 
    branches downsample the input through average pooling layers, process the downsampled data through 3x3 convolutional layers, and then 
    upsample it through transpose convolutional layers. Next, the outputs of all branches are concatenated, and a 1x1 convolutional layer is 
    applied to form the main path output. The branch path processes the input through a 1x1 convolutional layer to match the number of channels 
    of the main path. Finally, the main path and branch path outputs are fused together through addition. The final output is completed through 
    a fully connected layer for 10-class classification.
    """
    
    # Define the input shape
    input_shape = (32, 32, 3)
    
    # Define the input layer
    inputs = Input(shape=input_shape)
    
    # Main Path
    # Apply a 1x1 convolutional layer to the input
    x = Conv2D(64, (1, 1), activation='relu')(inputs)
    
    # Split into three branches
    # Branch 1: Extract local features through a 3x3 convolutional layer
    x1 = Conv2D(64, (3, 3), activation='relu')(x)
    
    # Branch 2: Downsample the input through average pooling layer, process the downsampled data through a 3x3 convolutional layer, 
    # and then upsample it through transpose convolutional layer
    x2 = Conv2D(64, (3, 3), activation='relu')(MaxPooling2D((2, 2))(x))
    x2 = Conv2D(64, (3, 3), activation='relu')(UpSampling2D((2, 2))(x2))
    
    # Branch 3: Downsample the input through average pooling layer, process the downsampled data through a 3x3 convolutional layer, 
    # and then upsample it through transpose convolutional layer
    x3 = Conv2D(64, (3, 3), activation='relu')(MaxPooling2D((2, 2))(x))
    x3 = Conv2D(64, (3, 3), activation='relu')(UpSampling2D((2, 2))(x3))
    
    # Concatenate the outputs of all branches and apply a 1x1 convolutional layer to form the main path output
    main_output = Concatenate()([x1, x2, x3])
    main_output = Conv2D(64, (1, 1), activation='relu')(main_output)
    
    # Branch Path
    # Process the input through a 1x1 convolutional layer to match the number of channels of the main path
    branch_output = Conv2D(64, (1, 1), activation='relu')(inputs)
    
    # Fuse the main path and branch path outputs together through addition
    output = Add()([main_output, branch_output])
    
    # Flatten the output
    output = Flatten()(output)
    
    # Complete the final output through a fully connected layer for 10-class classification
    outputs = Dense(10, activation='softmax')(output)
    
    # Define the model
    model = Model(inputs=inputs, outputs=outputs)
    
    return model