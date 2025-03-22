import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, ReLU, Add, AveragePooling2D, Dense

def basic_block(x, filters):
    # Main path
    main_path = Conv2D(filters, (3, 3), padding='same')(x)
    main_path = BatchNormalization()(main_path)
    main_path = ReLU()(main_path)
    
    # Branch path
    branch_path = Conv2D(filters, (3, 3), padding='same')(x)
    branch_path = BatchNormalization()(branch_path)
    
    # Addition
    output = Add()([main_path, branch_path])
    output = ReLU()(output)
    
    return output

def dl_model():
    inputs = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 with 3 channels
    
    # Initial convolution
    x = Conv2D(16, (3, 3), padding='same')(inputs)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    
    # First level: one basic block
    x = basic_block(x, 16)
    
    # Second level: two residual blocks
    for _ in range(2):
        x = basic_block(x, 16)
    
    # Third level: capture features from initial convolution output
    global_branch = Conv2D(16, (1, 1), padding='same')(inputs)
    global_branch = BatchNormalization()(global_branch)
    global_branch = ReLU()(global_branch)
    
    # Add global branch to second level output
    x = Add()([x, global_branch])
    
    # Global average pooling and dense layer for classification
    x = AveragePooling2D((8, 8))(x)
    x = Flatten()(x)
    outputs = Dense(10, activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    
    return model

# Example usage:
# model = dl_model()
# model.summary()