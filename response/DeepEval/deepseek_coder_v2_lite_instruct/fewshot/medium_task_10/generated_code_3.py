import keras
from keras.layers import Input, Conv2D, BatchNormalization, ReLU, Add, AveragePooling2D, Flatten, Dense

def basic_block(input_tensor, filters):
    # Main path
    x = Conv2D(filters=filters, kernel_size=(3, 3), strides=(1, 1), padding='same')(input_tensor)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    
    # Branch path
    branch = Conv2D(filters=filters, kernel_size=(3, 3), strides=(1, 1), padding='same')(input_tensor)
    
    # Addition
    output_tensor = Add()([x, branch])
    output_tensor = ReLU()(output_tensor)
    
    return output_tensor

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # Initial convolution layer
    x = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same')(input_layer)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    
    # First level of residual blocks
    x = basic_block(x, filters=16)
    x = basic_block(x, filters=16)
    
    # Second level of residual blocks
    x = basic_block(x, filters=32)
    x = basic_block(x, filters=32)
    
    # Third level of residual blocks
    global_branch = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
    x = basic_block(x, filters=64)
    x = Add()([x, global_branch])
    
    # Global average pooling and dense layer
    x = AveragePooling2D(pool_size=(4, 4))(x)
    x = Flatten()(x)
    output_layer = Dense(units=10, activation='softmax')(x)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model