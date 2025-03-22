import keras
import tensorflow as tf
from keras.layers import Input, AveragePooling2D, Lambda, Flatten, Concatenate, DepthwiseConv2D, Dense, Reshape

def dl_model():

    input_layer = Input(shape=(28, 28, 1))

    # First block with average pooling layers
    def block_1(input_tensor):
        # Average pooling with different scales
        avg_pool1 = AveragePooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(input_tensor)
        flatten1 = Flatten()(avg_pool1)
        
        avg_pool2 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(input_tensor)
        flatten2 = Flatten()(avg_pool2)
        
        avg_pool3 = AveragePooling2D(pool_size=(4, 4), strides=(4, 4), padding='same')(input_tensor)
        flatten3 = Flatten()(avg_pool3)
        
        # Concatenate the flattened outputs
        output_tensor = Concatenate()([flatten1, flatten2, flatten3])
        return output_tensor

    # Second block with depthwise separable convolutions
    def block_2(input_tensor):
        # Splitting the input into 4 groups
        inputs_groups = Lambda(lambda x: tf.split(value=x, num_or_size_splits=4, axis=-1))(input_tensor)
        
        # Depthwise separable convolutions with different kernel sizes
        conv1 = DepthwiseConv2D(kernel_size=(1, 1), padding='same', activation='relu')(inputs_groups[0])
        conv2 = DepthwiseConv2D(kernel_size=(3, 3), padding='same', activation='relu')(inputs_groups[1])
        conv3 = DepthwiseConv2D(kernel_size=(5, 5), padding='same', activation='relu')(inputs_groups[2])
        conv4 = DepthwiseConv2D(kernel_size=(7, 7), padding='same', activation='relu')(inputs_groups[3])
        
        # Concatenate the outputs from the depthwise separable convolutions
        output_tensor = Concatenate()([conv1, conv2, conv3, conv4])
        return output_tensor

    # Process through the first block
    block1_output = block_1(input_tensor=input_layer)
    
    # Fully connected layer followed by reshape
    dense = Dense(units=64, activation='relu')(block1_output)
    reshaped = Reshape(target_shape=(4, 4, 4))(dense)
    
    # Process through the second block
    block2_output = block_2(input_tensor=reshaped)

    # Flatten the output and final classification layer
    flatten = Flatten()(block2_output)
    output_layer = Dense(units=10, activation='softmax')(flatten)

    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model