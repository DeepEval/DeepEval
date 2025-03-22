import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, ReLU, Add, GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model

def basic_block(x, filters, kernel_size=3, stride=1):
    # Main path
    conv1 = Conv2D(filters, kernel_size=kernel_size, strides=stride, padding='same')(x)
    bn1 = BatchNormalization()(conv1)
    relu1 = ReLU()(bn1)
    
    # Branch (identity)
    branch = x
    
    # Add the paths
    out = Add()([relu1, branch])
    
    return out

def residual_block(x, filters, kernel_size=3, stride=1):
    # Main path with basic block
    main_path = basic_block(x, filters, kernel_size, stride)
    
    # Independent convolutional branch
    branch_path = Conv2D(filters, kernel_size=1, strides=stride, padding='same')(x)
    
    # Add the paths
    out = Add()([main_path, branch_path])
    
    return out

def dl_model():
    input_shape = (32, 32, 3)
    num_classes = 10
    
    inputs = Input(shape=input_shape)
    
    # Initial convolution to adjust input feature dimensionality to 16
    x = Conv2D(16, (3, 3), padding='same', activation='relu')(inputs)
    
    # First level with a basic block
    x = basic_block(x, 16)
    
    # Second level with two residual blocks
    x = residual_block(x, 32)
    x = residual_block(x, 32)
    
    # Third level with global branch
    global_branch = Conv2D(32, (1, 1), padding='same')(x)
    
    # Output from the second level added to global branch
    x = Add()([x, global_branch])
    
    # Global Average Pooling
    x = GlobalAveragePooling2D()(x)
    
    # Fully connected layer for classification
    outputs = Dense(num_classes, activation='softmax')(x)
    
    # Create the model
    model = Model(inputs=inputs, outputs=outputs)
    
    return model

# Example usage
model = dl_model()
model.summary()