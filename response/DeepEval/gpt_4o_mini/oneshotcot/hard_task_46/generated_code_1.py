import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, GlobalAveragePooling2D, Dense, Lambda
import tensorflow as tf

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # First block: Split input into three groups and apply separable convolutions
    split_inputs = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_layer)
    
    # Each path uses separable convolutions with different kernel sizes
    path1 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(split_inputs[0])
    path2 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(split_inputs[1])
    path3 = Conv2D(filters=32, kernel_size=(5, 5), padding='same', activation='relu')(split_inputs[2])
    
    # Concatenate the outputs from the three paths
    first_block_output = Concatenate()([path1, path2, path3])
    
    # Second block: Multiple branches for enhanced feature extraction
    conv_branch = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(first_block_output)
    
    # Series of layers: 1x1 convolution followed by two 3x3 convolutions
    path1_branch = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(first_block_output)
    path1_branch = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(path1_branch)
    path1_branch = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(path1_branch)
    
    # Max pooling branch
    max_pool_branch = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(first_block_output)
    
    # Concatenate outputs from the second block
    second_block_output = Concatenate()([conv_branch, path1_branch, max_pool_branch])
    
    # Global average pooling
    global_avg_pool = GlobalAveragePooling2D()(second_block_output)
    
    # Fully connected layer for classification
    output_layer = Dense(units=10, activation='softmax')(global_avg_pool)

    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model