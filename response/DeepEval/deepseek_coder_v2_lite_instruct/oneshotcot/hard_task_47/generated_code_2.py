import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Lambda, AveragePooling2D
from tensorflow.keras import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # First block
    def first_block(input_tensor):
        # Split the input into three groups
        split_1 = Lambda(lambda x: x[:, :, :, :10])(input_tensor)
        split_2 = Lambda(lambda x: x[:, :, :, 10:20])(input_tensor)
        split_3 = Lambda(lambda x: x[:, :, :, 20:])(input_tensor)
        
        # Apply depthwise separable convolution
        conv_1x1 = Conv2D(32, (1, 1), padding='same', activation='relu')(split_1)
        conv_3x3 = Conv2D(32, (3, 3), padding='same', activation='relu')(split_2)
        conv_5x5 = Conv2D(32, (5, 5), padding='same', activation='relu')(split_3)
        
        # Batch normalization
        conv_1x1 = BatchNormalization()(conv_1x1)
        conv_3x3 = BatchNormalization()(conv_3x3)
        conv_5x5 = BatchNormalization()(conv_5x5)
        
        # Concatenate the outputs
        concat_1 = Concatenate()([conv_1x1, conv_3x3, conv_5x5])
        return concat_1
    
    first_block_output = first_block(input_layer)

    # Second block
    def second_block(input_tensor):
        # Branch 1: 1x1 convolution followed by 1x1 convolution
        branch1 = Conv2D(32, (1, 1), padding='same', activation='relu')(input_tensor)
        branch1 = Conv2D(32, (1, 1), padding='same', activation='relu')(branch1)
        
        # Branch 2: 1x1 convolution, followed by 1x7 convolution, 7x1 convolution, and 3x3 convolution
        branch2 = Conv2D(32, (1, 1), padding='same', activation='relu')(input_tensor)
        branch2 = Conv2D(32, (1, 7), padding='same', activation='relu')(branch2)
        branch2 = Conv2D(32, (7, 1), padding='same', activation='relu')(branch2)
        branch2 = Conv2D(32, (3, 3), padding='same', activation='relu')(branch2)
        
        # Branch 3: Average pooling followed by 1x1 convolution
        branch3 = AveragePooling2D((3, 3), strides=1, padding='same')(input_tensor)
        branch3 = Conv2D(32, (1, 1), padding='same', activation='relu')(branch3)
        
        # Concatenate the outputs
        concat_2 = Concatenate()([branch1, branch2, branch3])
        return concat_2
    
    second_block_output = second_block(first_block_output)

    # Flatten the result
    flatten_layer = Flatten()(second_block_output)

    # Fully connected layers
    dense1 = Dense(128, activation='relu')(flatten_layer)
    dense2 = Dense(64, activation='relu')(dense1)
    output_layer = Dense(10, activation='softmax')(dense2)

    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model