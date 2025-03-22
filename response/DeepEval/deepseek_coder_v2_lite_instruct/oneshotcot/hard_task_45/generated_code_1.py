import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Lambda
from tensorflow.keras import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # First Block
    def first_block(input_tensor):
        # Split the input into three groups along the last dimension
        split_1 = Lambda(lambda x: x[:, :16, :, :])(input_tensor)
        split_2 = Lambda(lambda x: x[:, 16:32, :, :])(input_tensor)
        split_3 = Lambda(lambda x: x[:, 32:, :, :])(input_tensor)
        
        # Depthwise separable convolutions
        conv_1x1 = Conv2D(64, kernel_size=(1, 1), padding='same', activation='relu')(split_1)
        conv_3x3 = Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu', depthwise_kernel=True)(split_2)
        conv_5x5 = Conv2D(64, kernel_size=(5, 5), padding='same', activation='relu', depthwise_kernel=True)(split_3)
        
        # Concatenate the outputs
        concatenated = Concatenate()([conv_1x1, conv_3x3, conv_5x5])
        return concatenated
    
    first_block_output = first_block(input_layer)
    batch_norm_1 = BatchNormalization()(first_block_output)
    
    # Second Block
    def second_block(input_tensor):
        # Branch 1: 1x1 convolution
        branch_1 = Conv2D(64, kernel_size=(1, 1), padding='same', activation='relu')(input_tensor)
        
        # Branch 2: <1x1 convolution, 3x3 convolution>
        branch_2a = Conv2D(64, kernel_size=(1, 1), padding='same', activation='relu')(input_tensor)
        branch_2b = Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu')(branch_2a)
        
        # Branch 3: 3x3 convolution
        branch_3 = Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu')(input_tensor)
        
        # Branch 4: <max pooling, 1x1 convolution>
        branch_4a = MaxPooling2D(pool_size=(3, 3), strides=1, padding='same')(input_tensor)
        branch_4b = Conv2D(64, kernel_size=(1, 1), padding='same', activation='relu')(branch_4a)
        
        # Branch 5: <1x1 convolution, 3x3 convolution>
        branch_5a = Conv2D(64, kernel_size=(1, 1), padding='same', activation='relu')(input_tensor)
        branch_5b = Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu')(branch_5a)
        
        # Branch 6: <1x1 convolution, 3x3 convolution>
        branch_6a = Conv2D(64, kernel_size=(1, 1), padding='same', activation='relu')(input_tensor)
        branch_6b = Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu')(branch_6a)
        
        # Concatenate the outputs
        concatenated = Concatenate()([branch_1, branch_2b, branch_3, branch_4b, branch_5b, branch_6b])
        return concatenated
    
    second_block_output = second_block(batch_norm_1)
    batch_norm_2 = BatchNormalization()(second_block_output)
    flatten_layer = Flatten()(batch_norm_2)
    dense_layer = Dense(units=10, activation='softmax')(flatten_layer)
    
    model = Model(inputs=input_layer, outputs=dense_layer)
    
    return model