import keras
from keras.layers import Input, Conv2D, Lambda, Concatenate, Add, Flatten, Dense
import tensorflow as tf

def dl_model():
    # Input layer for CIFAR-10 images (32x32 with 3 color channels)
    input_layer = Input(shape=(32, 32, 3))
    
    # Main path
    def main_path(input_tensor):
        # Splitting the input into three parts along the channel dimension
        splits = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_tensor)
        
        # Apply different convolutions on each split
        conv1 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(splits[0])
        conv2 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(splits[1])
        conv3 = Conv2D(filters=32, kernel_size=(5, 5), padding='same', activation='relu')(splits[2])
        
        # Concatenate the results
        concatenated = Concatenate()([conv1, conv2, conv3])
        return concatenated
    
    # Branch path
    def branch_path(input_tensor):
        # Apply 1x1 convolution to align channels
        conv_branch = Conv2D(filters=96, kernel_size=(1, 1), padding='same', activation='relu')(input_tensor)
        return conv_branch
    
    # Obtain outputs from the main and branch paths
    main_output = main_path(input_layer)
    branch_output = branch_path(input_layer)
    
    # Add the outputs from the main and branch paths
    fused_features = Add()([main_output, branch_output])
    
    # Flatten the result and pass through fully connected layers
    flatten_layer = Flatten()(fused_features)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)
    
    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model