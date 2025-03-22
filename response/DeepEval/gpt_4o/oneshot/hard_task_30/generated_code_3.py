import keras
from keras.layers import Input, Conv2D, DepthwiseConv2D, Add, Concatenate, Flatten, Dense, Lambda
import tensorflow as tf

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # First block: Dual-path structure
    # Main path
    main_path = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    main_path = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(main_path)
    main_path = Conv2D(filters=3, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(main_path)
    
    # Branch path
    branch_path = input_layer
    
    # Combine paths
    combined_output = Add()([main_path, branch_path])
    
    # Second block: Split channels and depthwise separable convolution
    def split_and_depthwise_conv(input_tensor):
        # Split the channels into 3 groups
        splits = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_tensor)

        # Apply depthwise separable convolutions with different kernel sizes
        path1 = DepthwiseConv2D(kernel_size=(1, 1), padding='same', activation='relu')(splits[0])
        path2 = DepthwiseConv2D(kernel_size=(3, 3), padding='same', activation='relu')(splits[1])
        path3 = DepthwiseConv2D(kernel_size=(5, 5), padding='same', activation='relu')(splits[2])

        # Concatenate the outputs
        output_tensor = Concatenate()([path1, path2, path3])

        return output_tensor
    
    block2_output = split_and_depthwise_conv(combined_output)
    
    # Fully connected layers
    flatten_layer = Flatten()(block2_output)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model