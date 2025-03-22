import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Lambda
from tensorflow.keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # First block
    def first_block(input_tensor):
        # Depthwise separable convolutions
        conv1x1 = Conv2D(64, (1, 1), padding='same', activation='relu')(input_tensor)
        conv3x3 = Conv2D(64, (3, 3), padding='same', activation='relu')(input_tensor)
        conv5x5 = Conv2D(64, (5, 5), padding='same', activation='relu')(input_tensor)
        
        # Split into three groups
        split_layers = Lambda(lambda x: tf.split(x, 3, axis=-1))(input_tensor)
        depthwise_conv1x1 = Conv2D(64, (1, 1), padding='same', activation='relu')(split_layers[0])
        depthwise_conv3x3 = Conv2D(64, (3, 3), padding='same', activation='relu')(split_layers[1])
        depthwise_conv5x5 = Conv2D(64, (5, 5), padding='same', activation='relu')(split_layers[2])
        
        # Concatenate outputs
        concatenated = Concatenate()([depthwise_conv1x1, depthwise_conv3x3, depthwise_conv5x5])
        return concatenated
    
    first_block_output = first_block(input_layer)
    batch_norm1 = BatchNormalization()(first_block_output)

    # Second block
    def second_block(input_tensor):
        # Branch 1: 1x1 convolution
        branch1 = Conv2D(64, (1, 1), padding='same', activation='relu')(input_tensor)
        
        # Branch 2: 1x1 convolution followed by 3x3 convolution
        branch2a = Conv2D(64, (1, 1), padding='same', activation='relu')(input_tensor)
        branch2b = Conv2D(64, (3, 3), padding='same', activation='relu')(branch2a)
        
        # Branch 3: 1x1 convolution followed by 3x3 convolution
        branch3a = Conv2D(64, (1, 1), padding='same', activation='relu')(input_tensor)
        branch3b = Conv2D(64, (3, 3), padding='same', activation='relu')(branch3a)
        
        # Branch 4: max pooling followed by 1x1 convolution
        branch4a = MaxPooling2D((2, 2), strides=2)(input_tensor)
        branch4b = Conv2D(64, (1, 1), padding='same', activation='relu')(branch4a)
        
        # Concatenate outputs
        concatenated = Concatenate()([branch1, branch2b, branch3b, branch4b])
        return concatenated
    
    second_block_output = second_block(batch_norm1)
    batch_norm2 = BatchNormalization()(second_block_output)
    flatten_layer = Flatten()(batch_norm2)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = Model(inputs=input_layer, outputs=output_layer)

    return model