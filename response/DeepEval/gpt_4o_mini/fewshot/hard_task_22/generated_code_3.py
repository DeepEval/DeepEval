import keras
import tensorflow as tf
from keras.layers import Input, Lambda, Concatenate, Flatten, Dense, DepthwiseConv2D, Add, Conv2D

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Main path with multi-scale feature extraction
    def main_path(input_tensor):
        # Split the input tensor into three groups along the channel axis
        inputs_groups = Lambda(lambda x: tf.split(value=x, num_or_size_splits=3, axis=-1))(input_tensor)
        
        # Apply depthwise separable convolutions with varying kernel sizes
        conv1 = DepthwiseConv2D(kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(inputs_groups[0])
        conv2 = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(inputs_groups[1])
        conv3 = DepthwiseConv2D(kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(inputs_groups[2])
        
        # Concatenate outputs from the three groups
        output_tensor = Concatenate()([conv1, conv2, conv3])
        return output_tensor

    # Branch path to align channel dimensions
    def branch_path(input_tensor):
        return Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)

    # Get outputs from both paths
    main_path_output = main_path(input_layer)
    branch_path_output = branch_path(input_layer)

    # Fuse outputs from both paths
    fused_output = Add()([main_path_output, branch_path_output])

    # Flatten and pass through fully connected layers
    flatten_layer = Flatten()(fused_output)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model