import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, GlobalAveragePooling2D, Dense, Multiply, Concatenate, Flatten

def same_block(input_layer):
    # Global Average Pooling
    gap = GlobalAveragePooling2D(name='gap')(input_layer)
    
    # Fully connected layers
    fc1 = Dense(128, activation='relu', name='fc1')(gap)
    fc2 = Dense(32, activation='relu', name='fc2')(fc1)
    
    # Reshape to match input shape
    reshape = tf.expand_dims(fc2, axis=1)
    reshape = tf.expand_dims(reshape, axis=1)
    
    # Element-wise multiplication
    multiplied = Multiply(name='multiplied')([input_layer, reshape])
    
    return multiplied

def dl_model():
    # Input layer
    inputs = Input(shape=(32, 32, 3), name='input_layer')
    
    # First branch
    branch1 = Conv2D(32, (3, 3), activation='relu', name='branch1_conv1')(inputs)
    branch1 = Conv2D(32, (3, 3), activation='relu', name='branch1_conv2')(branch1)
    branch1 = same_block(branch1)
    
    # Second branch
    branch2 = Conv2D(32, (3, 3), activation='relu', name='branch2_conv1')(inputs)
    branch2 = Conv2D(32, (3, 3), activation='relu', name='branch2_conv2')(branch2)
    branch2 = same_block(branch2)
    
    # Concatenate outputs
    concatenated = Concatenate(name='concatenate')([branch1, branch2])
    
    # Flatten and fully connected layer
    flattened = Flatten(name='flatten')(concatenated)
    outputs = Dense(10, activation='softmax', name='output_layer')(flattened)
    
    # Create model
    model = Model(inputs=inputs, outputs=outputs)
    
    return model

# Example usage:
# model = dl_model()
# model.summary()