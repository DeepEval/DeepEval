import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, ReLU, Add, AveragePooling2D, Flatten, Dense
from tensorflow.keras.models import Model

def basic_block(x, filters, kernel_size=3, stride=1):
    # Main path
    conv = Conv2D(filters, kernel_size=kernel_size, strides=stride, padding='same')(x)
    bn = BatchNormalization()(conv)
    act = ReLU()(bn)
    
    # Short path (branch)
    shortcut = x
    
    # Combine main path and shortcut
    out = Add()([act, shortcut])
    return out

def dl_model():
    # Input layer
    inputs = Input(shape=(32, 32, 3))
    
    # Adjust input feature dimensionality to 16
    x = Conv2D(16, kernel_size=3, padding='same', activation='relu')(inputs)
    
    # Level 1: One basic block
    x = basic_block(x, 16)
    
    # Level 2: Two residual blocks
    # First block
    main_path_1 = basic_block(x, 16)
    branch_1 = Conv2D(16, kernel_size=3, padding='same')(x)
    x1 = Add()([main_path_1, branch_1])
    
    # Second block
    main_path_2 = basic_block(x1, 16)
    branch_2 = Conv2D(16, kernel_size=3, padding='same')(x1)
    x2 = Add()([main_path_2, branch_2])
    
    # Level 3: Global branch
    global_branch = Conv2D(16, kernel_size=3, padding='same')(x)
    x3 = Add()([x2, global_branch])
    
    # Global average pooling and fully connected layer
    x3 = AveragePooling2D(pool_size=(8, 8))(x3)
    x3 = Flatten()(x3)
    outputs = Dense(10, activation='softmax')(x3)
    
    # Model
    model = Model(inputs, outputs)
    
    return model

# Example usage
model = dl_model()
model.summary()