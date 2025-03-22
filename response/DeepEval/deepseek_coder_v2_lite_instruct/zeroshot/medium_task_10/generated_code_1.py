import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, ReLU, Add, AveragePooling2D, Flatten, Dense
from tensorflow.keras.models import Model

def basic_block(x, filters):
    # Main path
    main_path = Conv2D(filters, (3, 3), padding='same')(x)
    main_path = BatchNormalization()(main_path)
    main_path = ReLU()(main_path)
    
    # Branch path
    branch_path = Conv2D(filters, (3, 3), padding='same')(x)
    branch_path = BatchNormalization()(branch_path)
    
    # Addition operation
    output = Add()([main_path, branch_path])
    output = ReLU()(output)
    
    return output

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Initial convolution
    x = Conv2D(16, (3, 3), padding='same')(input_layer)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    
    # First level of the residual structure
    x = basic_block(x, 16)
    
    # Second level of the residual structure
    x = basic_block(x, 16)
    x = basic_block(x, 16)
    
    # Third level of the residual structure
    global_branch = Conv2D(16, (3, 3), padding='same')(x)
    global_branch = BatchNormalization()(global_branch)
    global_branch = ReLU()(global_branch)
    
    # Add global branch to the second-level residual structure
    x = Add()([x, global_branch])
    
    # Global average pooling
    x = AveragePooling2D((8, 8))(x)
    
    # Flatten layer
    x = Flatten()(x)
    
    # Fully connected layer
    output_layer = Dense(10, activation='softmax')(x)
    
    # Create model
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Example usage
model = dl_model()
model.summary()