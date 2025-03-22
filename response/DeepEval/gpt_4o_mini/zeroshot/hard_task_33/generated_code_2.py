import tensorflow as tf
from tensorflow.keras import layers, models

def create_branch(input_tensor):
    # Block starts with a 1x1 convolution to increase dimensions
    x = layers.Conv2D(32, (1, 1), activation='relu')(input_tensor)
    
    # Depthwise separable convolution
    x = layers.SeparableConv2D(64, (3, 3), padding='same', activation='relu')(x)
    
    # 1x1 convolution to reduce dimensions
    x = layers.Conv2D(32, (1, 1), activation='relu')(x)
    
    # Adding the input to the output (residual connection)
    x = layers.add([x, input_tensor])
    
    return x

def dl_model():
    input_shape = (28, 28, 1)  # MNIST images are 28x28 pixels with 1 channel
    inputs = layers.Input(shape=input_shape)
    
    # Create three branches
    branch1 = create_branch(inputs)
    branch2 = create_branch(inputs)
    branch3 = create_branch(inputs)
    
    # Concatenate the outputs from the branches
    concatenated = layers.concatenate([branch1, branch2, branch3])
    
    # Flatten the concatenated output
    flattened = layers.Flatten()(concatenated)
    
    # Fully connected layer with softmax activation for classification
    outputs = layers.Dense(10, activation='softmax')(flattened)  # 10 classes for MNIST
    
    # Create the model
    model = models.Model(inputs=inputs, outputs=outputs)
    
    return model

# Example usage:
model = dl_model()
model.summary()  # To show the model architecture