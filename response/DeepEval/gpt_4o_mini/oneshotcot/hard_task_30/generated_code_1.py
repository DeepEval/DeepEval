import keras
from keras.layers import Input, Conv2D, Add, Lambda, DepthwiseConv2D, Concatenate, Flatten, Dense, BatchNormalization
import tensorflow as tf

def dl_model():
    
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images have shape 32x32 with 3 color channels

    # First Block
    # Main Path
    main_path = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    main_path = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(main_path)
    main_path = Conv2D(filters=3, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(main_path)

    # Branch Path
    branch_path = input_layer

    # Combine paths
    combined_path = Add()([main_path, branch_path])

    # Second Block
    def split_and_depthwise_conv(input_tensor):
        # Split the input tensor into 3 groups along the channel axis
        split_tensors = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_tensor)
        
        # Extract features using depthwise separable convolutions
        depthwise_path1 = DepthwiseConv2D(kernel_size=(1, 1), padding='same', activation='relu')(split_tensors[0])
        depthwise_path2 = DepthwiseConv2D(kernel_size=(3, 3), padding='same', activation='relu')(split_tensors[1])
        depthwise_path3 = DepthwiseConv2D(kernel_size=(5, 5), padding='same', activation='relu')(split_tensors[2])
        
        # Concatenate the outputs
        concatenated_output = Concatenate()([depthwise_path1, depthwise_path2, depthwise_path3])
        
        return concatenated_output

    block_output = split_and_depthwise_conv(combined_path)

    # Flatten the output
    flatten_layer = Flatten()(block_output)

    # Fully connected layers
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=10, activation='softmax')(dense1)  # Output layer for 10 classes

    # Create the model
    model = keras.Model(inputs=input_layer, outputs=dense2)

    return model