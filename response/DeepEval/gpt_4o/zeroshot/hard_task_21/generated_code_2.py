from tensorflow.keras.layers import Input, Lambda, DepthwiseConv2D, Conv2D, Concatenate, Add, Flatten, Dense
from tensorflow.keras.models import Model
import tensorflow as tf

def dl_model():
    input_shape = (32, 32, 3)  # CIFAR-10 images are 32x32x3

    # Define the input layer
    inputs = Input(shape=input_shape)
    
    # Split the input into three groups along the channel dimension
    split_channels = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(inputs)
    
    # Main path processing with depthwise separable convolutions
    # First branch with 1x1 convolution
    conv1x1 = DepthwiseConv2D((1, 1), padding='same', activation='relu')(split_channels[0])
    
    # Second branch with 3x3 convolution
    conv3x3 = DepthwiseConv2D((3, 3), padding='same', activation='relu')(split_channels[1])
    
    # Third branch with 5x5 convolution
    conv5x5 = DepthwiseConv2D((5, 5), padding='same', activation='relu')(split_channels[2])
    
    # Concatenate the outputs from all three branches
    main_path_output = Concatenate()([conv1x1, conv3x3, conv5x5])
    
    # Branch path with a 1x1 convolution to align channels
    branch_path_output = Conv2D(main_path_output.shape[-1], (1, 1), padding='same', activation='relu')(inputs)
    
    # Add the outputs from the main and branch paths
    combined_output = Add()([main_path_output, branch_path_output])
    
    # Flatten and fully connected layers for classification
    flatten = Flatten()(combined_output)
    fc1 = Dense(512, activation='relu')(flatten)
    fc2 = Dense(10, activation='softmax')(fc1)
    
    # Create the model
    model = Model(inputs=inputs, outputs=fc2)
    
    return model