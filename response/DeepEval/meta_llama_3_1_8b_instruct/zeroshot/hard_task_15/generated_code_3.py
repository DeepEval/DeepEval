# Import necessary packages
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, GlobalAveragePooling2D, Add, Reshape, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K

def dl_model():
    """
    This function constructs a deep learning model for image classification using the CIFAR-10 dataset.
    
    The model consists of a main path and a branch path. In the main path, the input features undergo 
    global average pooling to extract global information from the feature map. The pooled features are 
    then passed through two fully connected layers to generate weights with the same number of channels 
    as the input layer. These weights are reshaped to match the input layer's shape and multiplied 
    element-wise with the input feature map. The branch path is directly connected to the input layer 
    without any modification. The outputs from both the main and branch paths are added together. 
    The combined result is then fed through two fully connected layers to produce the final probability 
    distribution.

    Parameters:
    None

    Returns:
    model (Model): The constructed deep learning model.
    """

    # Define the input shape
    input_shape = (32, 32, 3)

    # Define the input layer
    inputs = Input(shape=input_shape, name='input')

    # Main path
    x = Conv2D(32, (3, 3), activation='relu', name='conv1')(inputs)
    x = MaxPooling2D((2, 2), name='pool1')(x)
    x = Conv2D(64, (3, 3), activation='relu', name='conv2')(x)
    x = MaxPooling2D((2, 2), name='pool2')(x)
    x = Conv2D(128, (3, 3), activation='relu', name='conv3')(x)
    x = MaxPooling2D((2, 2), name='pool3')(x)
    
    # Global average pooling
    x = GlobalAveragePooling2D(name='avg_pool')(x)
    
    # Two fully connected layers
    x = Dense(128, activation='relu', name='fc1')(x)
    weights = Dense(3*32*32, activation='linear', name='weights')(x)
    weights = Reshape((32, 32, 3))(weights)
    
    # Element-wise multiplication
    x = Lambda(lambda x: x[0] * x[1])([inputs, weights])
    
    # Branch path
    y = inputs
    
    # Add the outputs from both paths
    x = Add()([x, y])
    
    # Two fully connected layers
    x = Dense(128, activation='relu', name='fc2')(x)
    outputs = Dense(10, activation='softmax', name='output')(x)

    # Define the model
    model = Model(inputs=inputs, outputs=outputs)

    return model