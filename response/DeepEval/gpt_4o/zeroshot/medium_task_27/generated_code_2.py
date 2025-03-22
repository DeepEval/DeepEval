import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, GlobalAveragePooling2D, Dense, Multiply, Add
from tensorflow.keras.models import Model

def dl_model():
    # Define the input shape for CIFAR-10 images
    input_shape = (32, 32, 3)
    
    # Input layer
    inputs = Input(shape=input_shape)
    
    # First branch: Convolution with 3x3 kernels
    branch1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    
    # Second branch: Convolution with 5x5 kernels
    branch2 = Conv2D(32, (5, 5), activation='relu', padding='same')(inputs)
    
    # Combine both branches through addition
    combined = Add()([branch1, branch2])
    
    # Global average pooling
    global_pool = GlobalAveragePooling2D()(combined)
    
    # Fully connected layer to generate attention weights
    attention_weights1 = Dense(32, activation='softmax')(global_pool)
    attention_weights2 = Dense(32, activation='softmax')(global_pool)
    
    # Multiply each branch with its corresponding attention weights
    weighted_branch1 = Multiply()([branch1, attention_weights1])
    weighted_branch2 = Multiply()([branch2, attention_weights2])
    
    # Add the weighted branches to produce the final weighted output
    weighted_output = Add()([weighted_branch1, weighted_branch2])
    
    # Fully connected layer to deliver the probability distribution across the 10 classes
    output = Dense(10, activation='softmax')(GlobalAveragePooling2D()(weighted_output))
    
    # Construct the model
    model = Model(inputs=inputs, outputs=output)
    
    return model

# Example usage:
# model = dl_model()
# model.summary()