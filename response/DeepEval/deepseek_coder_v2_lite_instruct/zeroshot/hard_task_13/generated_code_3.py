import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, GlobalAveragePooling2D, Dense, Reshape, Multiply

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # First block: four parallel branches
    branch1x1 = Conv2D(64, (1, 1), activation='relu')(input_layer)
    branch3x3 = Conv2D(64, (3, 3), padding='same', activation='relu')(input_layer)
    branch5x5 = Conv2D(64, (5, 5), padding='same', activation='relu')(input_layer)
    branch_pool = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(input_layer)
    
    # Concatenate the outputs of the four branches
    combined = Concatenate(axis=-1)([branch1x1, branch3x3, branch5x5, branch_pool])
    
    # Second block: reduce dimensionality using global average pooling
    gap = GlobalAveragePooling2D()(combined)
    
    # Pass through two fully connected layers
    fc1 = Dense(128, activation='relu')(gap)
    fc2 = Dense(128, activation='relu')(fc1)
    
    # Reshape the output to match the input shape
    reshaped = Reshape((32, 32, 128))(fc2)
    
    # Multiply the reshaped output with the input feature map
    multiplied = Multiply()([input_layer, reshaped])
    
    # Final fully connected layer for output
    output_layer = Dense(10, activation='softmax')(multiplied)
    
    # Define the model
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Example usage
model = dl_model()
model.summary()