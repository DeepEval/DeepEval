from keras.layers import Input, Conv2D, Dense, Flatten, Concatenate, Add
from keras.models import Model
import keras

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # First branch: 1x1 convolution
    branch1 = Conv2D(32, (1, 1), activation='relu', padding='same')(input_layer)
    
    # Second branch: 1x1 convolution followed by a 3x3 convolution
    branch2 = Conv2D(32, (1, 1), activation='relu', padding='same')(input_layer)
    branch2 = Conv2D(32, (3, 3), activation='relu', padding='same')(branch2)
    
    # Third branch: 1x1 convolution followed by two 3x3 convolutions
    branch3 = Conv2D(32, (1, 1), activation='relu', padding='same')(input_layer)
    branch3 = Conv2D(32, (3, 3), activation='relu', padding='same')(branch3)
    branch3 = Conv2D(32, (3, 3), activation='relu', padding='same')(branch3)
    
    # Concatenate branches
    concatenated = Concatenate()([branch1, branch2, branch3])
    
    # 1x1 convolution to adjust dimensions
    adjusted = Conv2D(3, (1, 1), activation='relu', padding='same')(concatenated)
    
    # Add shortcut connection (element-wise addition)
    fused = Add()([adjusted, input_layer])
    
    # Classification layers
    flat = Flatten()(fused)
    dense1 = Dense(128, activation='relu')(flat)
    dense2 = Dense(64, activation='relu')(dense1)
    output_layer = Dense(10, activation='softmax')(dense2)
    
    # Construct model
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Instantiate the model
model = dl_model()

# Display the model architecture
model.summary()