import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Flatten, Dense

def dl_model():
    input_shape = (32, 32, 3)
    
    # Main path
    inputs = Input(shape=input_shape)
    
    # Initial 1x1 convolutional layer
    x = Conv2D(32, (1, 1), activation='relu')(inputs)
    
    # Branch 1: 3x3 convolutional layer
    branch1 = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    
    # Branch 2: Max pooling, 3x3 convolutional layer, and upsampling
    branch2 = MaxPooling2D((2, 2))(x)
    branch2 = Conv2D(32, (3, 3), activation='relu', padding='same')(branch2)
    branch2 = UpSampling2D((2, 2))(branch2)
    
    # Branch 3: Max pooling, 3x3 convolutional layer, and upsampling
    branch3 = MaxPooling2D((2, 2))(x)
    branch3 = Conv2D(32, (3, 3), activation='relu', padding='same')(branch3)
    branch3 = UpSampling2D((2, 2))(branch3)
    
    # Concatenate outputs from all branches
    combined = concatenate([branch1, branch2, branch3])
    
    # Final 1x1 convolutional layer
    main_output = Conv2D(64, (1, 1), activation='relu')(combined)
    
    # Branch path
    branch_inputs = Input(shape=input_shape)
    branch_path = Conv2D(64, (1, 1), activation='relu')(branch_inputs)
    
    # Add outputs from main path and branch path
    added = concatenate([main_output, branch_path])
    
    # Flatten and pass through fully connected layers
    x = Flatten()(added)
    x = Dense(128, activation='relu')(x)
    outputs = Dense(10, activation='softmax')(x)
    
    # Create the model
    model = Model(inputs=[inputs, branch_inputs], outputs=outputs)
    
    return model

# Example usage
model = dl_model()
model.summary()