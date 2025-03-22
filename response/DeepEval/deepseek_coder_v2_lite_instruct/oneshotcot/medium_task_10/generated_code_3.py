import keras
from keras.layers import Input, Conv2D, BatchNormalization, ReLU, Add, AveragePooling2D, Flatten, Dense
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # Adjusted input shape for CIFAR-10

    # Initial convolution to adjust the input feature dimensionality
    x = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same')(input_layer)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    # Define the basic block
    def basic_block(input_tensor, filters):
        # Main path
        main_path = Conv2D(filters=filters, kernel_size=(3, 3), strides=(1, 1), padding='same')(input_tensor)
        main_path = BatchNormalization()(main_path)
        main_path = ReLU()(main_path)
        
        # Branch path (identity connection)
        branch_path = Conv2D(filters=filters, kernel_size=(3, 3), strides=(1, 1), padding='same')(input_tensor)
        
        # Addition of main path and branch path
        output_tensor = Add()([main_path, branch_path])
        output_tensor = ReLU()(output_tensor)
        
        return output_tensor

    # First level of the residual structure
    x = basic_block(x, filters=16)

    # Second level of the residual structure with two basic blocks
    x = basic_block(x, filters=16)
    x = basic_block(x, filters=16)

    # Third level of the residual structure
    global_branch = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same')(input_layer)
    global_branch = BatchNormalization()(global_branch)
    global_branch = ReLU()(global_branch)
    
    # Add the global branch to the second level residual structure
    x = Add()([x, global_branch])

    # Final processing
    x = AveragePooling2D(pool_size=(8, 8))(x)
    x = Flatten()(x)
    output_layer = Dense(units=10, activation='softmax')(x)

    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model