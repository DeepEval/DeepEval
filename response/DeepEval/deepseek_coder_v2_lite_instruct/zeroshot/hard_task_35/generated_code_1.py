import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, GlobalAveragePooling2D, Dense, Add, Multiply, Flatten, Concatenate

def same_block(inputs):
    # Apply Global Average Pooling
    gap = GlobalAveragePooling2D(name='gap')(inputs)
    
    # Pass through two fully connected layers
    fc1 = Dense(128, activation='relu', name='fc1')(gap)
    fc2 = Dense(inputs.shape[-1], name='fc2')(fc1)
    
    # Reshape the output to match the input shape
    reshape = tf.reshape(fc2, inputs.shape)
    
    # Element-wise multiplication with the input
    output = Multiply(name='multiplication')([inputs, reshape])
    
    return output

def dl_model():
    # Input layer
    inputs = Input(shape=(32, 32, 3))
    
    # First branch
    branch1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    branch1 = same_block(branch1)
    
    # Second branch
    branch2 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    branch2 = same_block(branch2)
    
    # Concatenate the outputs of both branches
    concatenated = Concatenate(name='concatenation')([branch1, branch2])
    
    # Flatten the concatenated output
    flattened = Flatten(name='flatten')(concatenated)
    
    # Fully connected layer
    outputs = Dense(10, activation='softmax', name='output')(flattened)
    
    # Create the model
    model = Model(inputs=inputs, outputs=outputs)
    
    return model

# Example usage:
# model = dl_model()
# model.summary()