import keras
from keras.layers import Input, Conv2D, BatchNormalization, ReLU, Add, AveragePooling2D, Flatten, Dense
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # Initial convolution to adjust the input feature dimensionality to 16
    conv1 = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same')(input_layer)
    conv1_bn = BatchNormalization()(conv1)
    conv1_relu = ReLU()(conv1_bn)
    
    # Define the basic block
    def basic_block(input_tensor, filters):
        x = Conv2D(filters=filters, kernel_size=(3, 3), strides=(1, 1), padding='same')(input_tensor)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Conv2D(filters=filters, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        
        # Branch path
        branch = Conv2D(filters=filters, kernel_size=(3, 3), strides=(1, 1), padding='same')(input_tensor)
        branch = BatchNormalization()(branch)
        branch = ReLU()(branch)
        
        # Add the main path and the branch path
        output_tensor = Add()([x, branch])
        output_tensor = ReLU()(output_tensor)
        return output_tensor
    
    # First level of the residual connection structure
    x = basic_block(conv1_relu, filters=32)
    
    # Second level of the residual connection structure
    y = basic_block(x, filters=64)
    y = basic_block(y, filters=64)
    
    # Third level of the residual connection structure
    global_branch = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same')(input_layer)
    global_branch = BatchNormalization()(global_branch)
    global_branch = ReLU()(global_branch)
    
    # Add the global branch to the second-level residual structure
    z = Add()([y, global_branch])
    z = ReLU()(z)
    
    # Average pooling and fully connected layer
    z = AveragePooling2D(pool_size=(4, 4))(z)
    z = Flatten()(z)
    output_layer = Dense(units=10, activation='softmax')(z)
    
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model