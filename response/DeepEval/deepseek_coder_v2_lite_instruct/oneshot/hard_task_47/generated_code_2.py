import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Lambda, DepthwiseConv2D
from tensorflow.keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # First block
    def first_block(x):
        # Split the input into three groups
        split_1 = Lambda(lambda tensor: tensor[:, :, :, :10])(x)
        split_2 = Lambda(lambda tensor: tensor[:, :, :, 10:20])(x)
        split_3 = Lambda(lambda tensor: tensor[:, :, :, 20:])(x)
        
        # Depthwise separable convolutions
        depthwise_1x1 = DepthwiseConv2D(kernel_size=(1, 1), padding='same', activation='relu')(split_1)
        depthwise_3x3 = DepthwiseConv2D(kernel_size=(3, 3), padding='same', activation='relu')(split_2)
        depthwise_5x5 = DepthwiseConv2D(kernel_size=(5, 5), padding='same', activation='relu')(split_3)
        
        # Batch normalization
        depthwise_1x1 = BatchNormalization()(depthwise_1x1)
        depthwise_3x3 = BatchNormalization()(depthwise_3x3)
        depthwise_5x5 = BatchNormalization()(depthwise_5x5)
        
        # Concatenate outputs
        concat = Concatenate()([depthwise_1x1, depthwise_3x3, depthwise_5x5])
        return concat
    
    first_block_output = first_block(input_layer)
    
    # Second block
    def second_block(x):
        # Branch 1: 1x1 convolution followed by another 1x1 convolution
        branch1 = Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(x)
        branch1 = Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(branch1)
        
        # Branch 2: 1x1 convolution followed by 1x7 and 7x1 convolutions, then 3x3 convolution
        branch2 = Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(x)
        branch2 = Conv2D(filters=64, kernel_size=(1, 7), activation='relu')(branch2)
        branch2 = Conv2D(filters=64, kernel_size=(7, 1), activation='relu')(branch2)
        branch2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(branch2)
        
        # Branch 3: Average pooling
        branch3 = tf.reduce_mean(x, axis=-1, keepdims=True)
        branch3 = tf.reduce_mean(branch3, axis=-2, keepdims=True)
        branch3 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(branch3)
        
        # Concatenate outputs
        concat = Concatenate()([branch1, branch2, branch3])
        return concat
    
    second_block_output = second_block(first_block_output)
    
    # Flatten and fully connected layers
    flatten = Flatten()(second_block_output)
    dense1 = Dense(units=256, activation='relu')(flatten)
    dense2 = Dense(units=10, activation='softmax')(dense1)
    
    model = Model(inputs=input_layer, outputs=dense2)
    
    return model

# Example usage:
# model = dl_model()
# model.summary()