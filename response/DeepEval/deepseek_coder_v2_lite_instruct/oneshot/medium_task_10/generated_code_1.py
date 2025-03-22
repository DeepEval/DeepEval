import keras
from keras.layers import Input, Conv2D, BatchNormalization, ReLU, Add, AveragePooling2D, Flatten, Dense

def basic_block(input_tensor, filters):
    x = Conv2D(filters=filters, kernel_size=(3, 3), padding='same')(input_tensor)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    
    x = Conv2D(filters=filters, kernel_size=(3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    
    x = Add()([x, input_tensor])
    x = ReLU()(x)
    return x

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # Initial convolution to adjust the input feature dimensionality
    x = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same')(input_layer)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    
    # First level of the residual structure
    residual1 = basic_block(x, filters=16)
    
    # Second level of the residual structure with two residual blocks
    residual2 = basic_block(residual1, filters=16)
    residual2 = basic_block(residual2, filters=16)
    
    # Third level capturing features from the initial convolution output
    global_branch = Conv2D(filters=16, kernel_size=(1, 1), padding='same')(x)
    global_branch = BatchNormalization()(global_branch)
    global_branch = ReLU()(global_branch)
    
    # Add the global branch to the second-level residual structure
    x = Add()([residual2, global_branch])
    x = ReLU()(x)
    
    # Average pooling and fully connected layer for classification
    x = AveragePooling2D(pool_size=(4, 4))(x)
    x = Flatten()(x)
    output_layer = Dense(units=10, activation='softmax')(x)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model