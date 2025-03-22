import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, Flatten, Dense

def dl_model():
    input_shape = (32, 32, 3)
    
    # Branch 1: 3x3 convolutions
    input_branch1 = Input(shape=input_shape)
    conv_branch1 = Conv2D(32, (3, 3), activation='relu')(input_branch1)
    pool_branch1 = MaxPooling2D((2, 2))(conv_branch1)
    
    # Branch 2: 1x1 convolutions followed by two 3x3 convolutions
    input_branch2 = Input(shape=input_shape)
    conv_branch2 = Conv2D(32, (1, 1), activation='relu')(input_branch2)
    conv_branch2 = Conv2D(32, (3, 3), activation='relu')(conv_branch2)
    pool_branch2 = MaxPooling2D((2, 2))(conv_branch2)
    
    # Branch 3: Max pooling
    input_branch3 = Input(shape=input_shape)
    pool_branch3 = MaxPooling2D((2, 2))(input_branch3)
    
    # Concatenate the outputs from the three branches
    merged = Concatenate()([pool_branch1, pool_branch2, pool_branch3])
    
    # Flatten the concatenated feature maps
    flattened = Flatten()(merged)
    
    # Fully connected layers
    fc1 = Dense(128, activation='relu')(flattened)
    output = Dense(10, activation='softmax')(fc1)
    
    # Create the model
    model = Model(inputs=[input_branch1, input_branch2, input_branch3], outputs=output)
    
    return model

# Example usage
model = dl_model()
model.summary()