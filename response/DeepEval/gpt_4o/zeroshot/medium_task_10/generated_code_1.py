from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, ReLU, Add, AveragePooling2D, Flatten, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import cifar10

def basic_block(input_tensor, filters):
    # Main path
    x = Conv2D(filters, (3, 3), padding='same', kernel_initializer='he_normal')(input_tensor)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    
    # Branch (direct connection)
    shortcut = input_tensor
    
    # Combine main path and branch
    x = Add()([x, shortcut])
    return x

def dl_model():
    # Input layer
    input_tensor = Input(shape=(32, 32, 3))
    
    # Initial convolution to adjust dimensionality
    x = Conv2D(16, (3, 3), padding='same', kernel_initializer='he_normal')(input_tensor)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    
    # Level 1
    x = basic_block(x, 16)
    
    # Level 2
    # First block
    main_path = basic_block(x, 16)
    branch = Conv2D(16, (3, 3), padding='same', kernel_initializer='he_normal')(x)
    branch = BatchNormalization()(branch)
    level_2_output_1 = Add()([main_path, branch])
    
    # Second block
    main_path = basic_block(level_2_output_1, 16)
    branch = Conv2D(16, (3, 3), padding='same', kernel_initializer='he_normal')(level_2_output_1)
    branch = BatchNormalization()(branch)
    level_2_output_2 = Add()([main_path, branch])
    
    # Level 3
    global_branch = Conv2D(16, (3, 3), padding='same', kernel_initializer='he_normal')(x)
    global_branch = BatchNormalization()(global_branch)
    level_3_output = Add()([level_2_output_2, global_branch])
    
    # Final layers
    x = AveragePooling2D(pool_size=(8, 8))(level_3_output)
    x = Flatten()(x)
    x = Dense(10, activation='softmax')(x)
    
    # Create model
    model = Model(inputs=input_tensor, outputs=x)
    
    return model

# Example usage:
# model = dl_model()
# model.summary()