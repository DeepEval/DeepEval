import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Add, Lambda, DepthwiseConv2D, Concatenate, BatchNormalization, Flatten, Dense
from keras.models import Model
import tensorflow as tf

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    def block1(x):
        # Main path
        main_path = Conv2D(32, (3, 3), padding='same', activation='relu')(x)
        main_path = Conv2D(32, (3, 3), padding='same', activation='relu')(main_path)
        # Branch path
        branch_path = Conv2D(32, (1, 1), padding='same', activation='relu')(x)
        # Add paths
        output_tensor = Add()([main_path, branch_path])
        return output_tensor
    
    # Apply block1 to input
    block1_output = block1(input_layer)
    block1_output = BatchNormalization()(block1_output)
    
    def block2(x):
        # Split input into three groups
        split_1 = Lambda(lambda z: z[:, :16, :16, :])(x)
        split_2 = Lambda(lambda z: z[:, 16:, :16, :])(x)
        split_3 = Lambda(lambda z: z[:, :16, 16:, :])(x)
        
        # Process each group with depthwise separable convolutions
        depthwise_1x1 = DepthwiseConv2D(kernel_size=(1, 1), padding='same', activation='relu')(split_1)
        depthwise_3x3 = DepthwiseConv2D(kernel_size=(3, 3), padding='same', activation='relu')(split_2)
        depthwise_5x5 = DepthwiseConv2D(kernel_size=(5, 5), padding='same', activation='relu')(split_3)
        
        # Concatenate outputs
        output_tensor = Concatenate()([depthwise_1x1, depthwise_3x3, depthwise_5x5])
        return output_tensor
    
    # Apply block2 to block1 output
    block2_output = block2(block1_output)
    block2_output = BatchNormalization()(block2_output)
    block2_output = Flatten()(block2_output)
    
    # Fully connected layers
    dense1 = Dense(128, activation='relu')(block2_output)
    output_layer = Dense(10, activation='softmax')(dense1)
    
    model = Model(inputs=input_layer, outputs=output_layer)
    return model

# Example usage:
# model = dl_model()
# model.summary()