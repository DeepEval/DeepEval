import keras
import tensorflow as tf
from keras.layers import Input, Lambda, DepthwiseConv2D, Concatenate, Add, Dense, Flatten

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 pixels with 3 color channels

    # Main Path
    def main_path(input_tensor):
        # Split the input tensor into three groups along the channel axis
        inputs_groups = Lambda(lambda x: tf.split(value=x, num_or_size_splits=3, axis=-1))(input_tensor)
        
        # Apply depthwise separable convolution with different kernel sizes
        conv1 = DepthwiseConv2D(kernel_size=(1, 1), padding='same', activation='relu')(inputs_groups[0])
        conv2 = DepthwiseConv2D(kernel_size=(3, 3), padding='same', activation='relu')(inputs_groups[1])
        conv3 = DepthwiseConv2D(kernel_size=(5, 5), padding='same', activation='relu')(inputs_groups[2])
        
        # Concatenate the outputs from the three convolutions
        output_tensor = Concatenate()([conv1, conv2, conv3])
        return output_tensor
    
    main_path_output = main_path(input_layer)

    # Branch Path
    branch_output = DepthwiseConv2D(kernel_size=(1, 1), padding='same', activation='relu')(input_layer)

    # Align the number of channels in branch path output to match main path output
    branch_output = Lambda(lambda x: tf.image.resize(x, (32, 32)))(branch_output)  # Ensure the size matches if necessary

    # Add the outputs from the main path and branch path
    combined_output = Add()([main_path_output, branch_output])

    # Fully connected layers for classification
    flatten_layer = Flatten()(combined_output)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)  # 10 classes for CIFAR-10

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model