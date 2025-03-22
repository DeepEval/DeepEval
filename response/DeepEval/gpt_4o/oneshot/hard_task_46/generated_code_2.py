import keras
from keras.layers import Input, Lambda, SeparableConv2D, Concatenate, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense
import tensorflow as tf

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # First Block: Channel-wise splitting and separable convolutions
    def split_and_separable_conv(input_tensor):
        # Split into 3 groups along the channel dimension
        splits = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_tensor)
        
        conv1 = SeparableConv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(splits[0])
        conv2 = SeparableConv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(splits[1])
        conv3 = SeparableConv2D(filters=32, kernel_size=(5, 5), padding='same', activation='relu')(splits[2])
        
        # Concatenate the outputs of the separable convolutions
        return Concatenate()([conv1, conv2, conv3])

    first_block_output = split_and_separable_conv(input_layer)
    
    # Second Block: Multiple branches for enhanced feature extraction
    def multi_branch_block(input_tensor):
        # Branch 1: A single 3x3 convolution
        branch1 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(input_tensor)
        
        # Branch 2: A series of layers - 1x1 conv followed by two 3x3 convs
        branch2 = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(input_tensor)
        branch2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(branch2)
        branch2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(branch2)
        
        # Branch 3: Max pooling
        branch3 = MaxPooling2D(pool_size=(3, 3), strides=1, padding='same')(input_tensor)
        
        # Concatenate the outputs from all branches
        return Concatenate()([branch1, branch2, branch3])

    second_block_output = multi_branch_block(first_block_output)
    
    # Global Average Pooling
    global_avg_pool = GlobalAveragePooling2D()(second_block_output)
    
    # Fully connected layer for classification
    output_layer = Dense(units=10, activation='softmax')(global_avg_pool)
    
    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model