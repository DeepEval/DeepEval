import tensorflow as tf
from tensorflow.keras import layers, models

def custom_block(input_tensor):
    # Global Average Pooling
    x = layers.GlobalAveragePooling2D()(input_tensor)
    
    # Fully Connected Layers
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dense(layers.Input(shape=(input_tensor.shape[-1],))(x))  # Output dim = number of channels
    weights = layers.Reshape((1, 1, input_tensor.shape[-1]))(x)  # Reshape to (1, 1, channels)
    
    # Element-wise multiplication
    output = layers.multiply([input_tensor, weights])
    
    return output

def dl_model():
    # Input Layer
    input_shape = (32, 32, 3)  # CIFAR-10 image shape
    inputs = layers.Input(shape=input_shape)
    
    # First branch
    branch1 = custom_block(inputs)
    
    # Second branch
    branch2 = custom_block(inputs)
    
    # Concatenate branches
    concatenated = layers.Concatenate()([branch1, branch2])
    
    # Flatten and Fully Connected Layer
    x = layers.Flatten()(concatenated)
    x = layers.Dense(256, activation='relu')(x)
    outputs = layers.Dense(10, activation='softmax')(x)  # 10 classes for CIFAR-10
    
    # Construct the model
    model = models.Model(inputs=inputs, outputs=outputs)
    
    return model

# Create the model
model = dl_model()
# Print the model summary
model.summary()