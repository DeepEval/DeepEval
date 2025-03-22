import keras
from keras.layers import Input, Lambda, SeparableConv2D, Concatenate, Add, Flatten, Dense
import tensorflow as tf

def dl_model():
    
    # Input layer with CIFAR-10 image shape
    input_layer = Input(shape=(32, 32, 3))
    
    # Splitting input into three groups along the channel dimension
    split_layer = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_layer)
    
    # Multi-scale feature extraction using separable convolutions
    branch1 = SeparableConv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(split_layer[0])
    branch2 = SeparableConv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(split_layer[1])
    branch3 = SeparableConv2D(filters=32, kernel_size=(5, 5), padding='same', activation='relu')(split_layer[2])
    
    # Concatenate the outputs from the three branches
    main_path_output = Concatenate()([branch1, branch2, branch3])
    
    # Branch path: 1x1 Convolution to align the channel numbers
    branch_path = SeparableConv2D(filters=96, kernel_size=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Fuse outputs from main path and branch path using addition
    fused_output = Add()([main_path_output, branch_path])
    
    # Flatten the fused output
    flattened_output = Flatten()(fused_output)
    
    # Fully connected layers for classification
    dense1 = Dense(units=128, activation='relu')(flattened_output)
    output_layer = Dense(units=10, activation='softmax')(dense1)
    
    # Build the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model