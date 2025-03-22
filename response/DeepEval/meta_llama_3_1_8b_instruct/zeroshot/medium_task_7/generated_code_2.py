# Import necessary packages
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Add, concatenate, Flatten, Dense

def dl_model():
    """
    This function constructs a deep learning model for image classification using the CIFAR-10 dataset.
    
    The model consists of three sequential convolutional layers, each processing the output of the previous layer.
    The outputs of the first two convolutional layers are added with the output of the third convolutional layer.
    Simultaneously, a separate convolutional layer processes the input directly.
    The added outputs from all paths are then passed through two fully connected layers for classification.
    """
    
    # Define input shape for CIFAR-10 dataset
    input_shape = (32, 32, 3)
    
    # Define the first convolutional path
    input_conv = Input(shape=input_shape)
    conv1 = Conv2D(32, (3, 3), activation='relu')(input_conv)
    conv1 = MaxPooling2D((2, 2))(conv1)
    conv1 = Conv2D(64, (3, 3), activation='relu')(conv1)
    conv1 = MaxPooling2D((2, 2))(conv1)
    conv1 = Conv2D(128, (3, 3), activation='relu')(conv1)
    
    # Define the second convolutional path
    input_conv2 = Input(shape=input_shape)
    conv2 = Conv2D(32, (3, 3), activation='relu')(input_conv2)
    conv2 = MaxPooling2D((2, 2))(conv2)
    conv2 = Conv2D(64, (3, 3), activation='relu')(conv2)
    
    # Define the third convolutional path
    input_conv3 = Input(shape=input_shape)
    conv3 = Conv2D(32, (3, 3), activation='relu')(input_conv3)
    conv3 = MaxPooling2D((2, 2))(conv3)
    conv3 = Conv2D(128, (3, 3), activation='relu')(conv3)
    
    # Define the separate convolutional layer
    conv4 = Conv2D(128, (3, 3), activation='relu')(input_conv)
    conv4 = MaxPooling2D((2, 2))(conv4)
    
    # Add outputs of the first two convolutional layers with the output of the third convolutional layer
    add_output = Add()([conv1, conv2, conv3])
    
    # Add the output of the separate convolutional layer
    add_output = Add()([add_output, conv4])
    
    # Flatten the output and pass it through two fully connected layers
    flatten_output = Flatten()(add_output)
    fc1 = Dense(128, activation='relu')(flatten_output)
    output = Dense(10, activation='softmax')(fc1)
    
    # Define the model
    model = Model(inputs=[input_conv, input_conv2, input_conv3], outputs=output)
    
    return model