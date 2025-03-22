import tensorflow as tf
from keras.layers import Input, Conv2D, SeparableConv2D, Lambda, Concatenate, Add, Flatten, Dense
from keras.models import Model

def dl_model():
    # Input layer for CIFAR-10 images
    input_layer = Input(shape=(32, 32, 3))
    
    # Main path: Split the input channels into three groups
    split_layers = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_layer)
    
    # Apply depthwise separable convolutions with different kernel sizes
    conv1x1 = SeparableConv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(split_layers[0])
    conv3x3 = SeparableConv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(split_layers[1])
    conv5x5 = SeparableConv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(split_layers[2])
    
    # Concatenate the outputs of the three convolutions
    main_path_output = Concatenate()([conv1x1, conv3x3, conv5x5])
    
    # Branch path: Use a 1x1 convolution to align with the main path
    branch_path_output = Conv2D(filters=96, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Combine the outputs of the main and branch paths
    combined_output = Add()([main_path_output, branch_path_output])
    
    # Flatten the combined output and add fully connected layers
    flatten_layer = Flatten()(combined_output)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)
    
    # Construct the model
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model