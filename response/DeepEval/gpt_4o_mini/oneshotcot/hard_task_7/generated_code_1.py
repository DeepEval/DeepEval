import keras
from keras.layers import Input, Conv2D, DepthwiseConv2D, Lambda, Concatenate, Flatten, Dense
import tensorflow as tf

def dl_model():
    
    input_layer = Input(shape=(28, 28, 1))
    
    # Initial convolutional layer with 32 kernels
    conv_initial = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    def block1(input_tensor):
        # Split the input tensor into two groups along the last dimension
        split_tensor = Lambda(lambda x: tf.split(x, num_or_size_splits=2, axis=-1))(input_tensor)
        
        # First group operations
        group1 = split_tensor[0]
        path1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(group1)
        path2 = DepthwiseConv2D(kernel_size=(3, 3), padding='same', activation='relu')(path1)
        path3 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(path2)

        # Second group (unmodified)
        group2 = split_tensor[1]
        
        # Concatenate both groups
        output_tensor = Concatenate()([path3, group2])
        
        return output_tensor
    
    block1_output = block1(input_tensor=conv_initial)
    
    def block2(input_tensor):
        # Obtain the shape of the input tensor
        shape = tf.shape(input_tensor)
        height, width, channels = shape[1], shape[2], shape[3]
        groups = 4
        channels_per_group = channels // groups
        
        # Reshape input into four groups
        reshaped_tensor = tf.reshape(input_tensor, (shape[0], height, width, groups, channels_per_group))
        
        # Permute the dimensions to achieve channel shuffling
        permuted_tensor = tf.transpose(reshaped_tensor, perm=(0, 1, 2, 4, 3))
        
        # Reshape back to original shape
        shuffled_tensor = tf.reshape(permuted_tensor, (shape[0], height, width, channels))
        
        return shuffled_tensor
    
    block2_output = block2(input_tensor=block1_output)
    
    # Flatten the output and pass through a fully connected layer
    flatten_layer = Flatten()(block2_output)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model