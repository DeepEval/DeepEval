import keras
from keras.models import Model
from keras.layers import Input, Conv2D, AveragePooling2D, Add, GlobalAveragePooling2D, Dense, Flatten

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Block 1: Feature extraction
    def block1(x):
        # First convolutional layer
        conv1 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(x)
        # Second convolutional layer
        conv2 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(conv1)
        # Average pooling
        avg_pool = AveragePooling2D(pool_size=(2, 2))(conv2)
        # Output of block 1
        return avg_pool
    
    block1_output = block1(input_layer)
    
    # Block 2: Refinement and classification
    def block2(x):
        # Global average pooling
        avg_pool = GlobalAveragePooling2D()(x)
        # Two fully connected layers
        fc1 = Dense(units=512, activation='relu')(avg_pool)
        # Output layer
        output = Dense(units=10, activation='softmax')(fc1)
        # Output of block 2
        return output
    
    block2_output = block2(block1_output)
    
    # Combine outputs of block 1 and block 2
    combined_output = Add()([block1_output, block2_output])
    
    # Flatten and output layer
    flatten = Flatten()(combined_output)
    output_layer = Dense(units=10, activation='softmax')(flatten)
    
    # Model
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Build the model
model = dl_model()
model.summary()