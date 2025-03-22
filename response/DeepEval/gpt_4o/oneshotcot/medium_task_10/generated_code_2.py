import keras
from keras.layers import Input, Conv2D, BatchNormalization, Activation, Add, GlobalAveragePooling2D, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # Initial Convolution to adjust input feature dimensionality
    initial_conv = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)

    def basic_block(input_tensor):
        # Main path
        x = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same')(input_tensor)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        
        # Adding input_tensor (branch) directly
        output_tensor = Add()([x, input_tensor])
        return output_tensor

    # First level
    level_1_output = basic_block(initial_conv)
    
    def residual_block(input_tensor):
        # Main path using basic block
        main_path = basic_block(input_tensor)
        
        # Branch with independent convolutional layer
        branch = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same')(input_tensor)
        
        # Combine main path and branch
        output_tensor = Add()([main_path, branch])
        return output_tensor

    # Second level with two residual blocks
    level_2_output = residual_block(level_1_output)
    level_2_output = residual_block(level_2_output)

    # Third level
    global_branch = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same')(initial_conv)
    
    # Combine second level output and global branch
    level_3_output = Add()([level_2_output, global_branch])

    # Average pooling and dense layer for classification
    avg_pool = GlobalAveragePooling2D()(level_3_output)
    output_layer = Dense(units=10, activation='softmax')(avg_pool)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model