import keras
from keras.layers import Input, Conv2D, DepthwiseConv2D, Reshape, Permute, Lambda, Concatenate, BatchNormalization, Flatten, Dense
from keras import backend as K
import tensorflow as tf

def dl_model():     

    input_layer = Input(shape=(28, 28, 1))

    # Initial convolutional layer
    conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='sigmoid')(input_layer)

    def block1(input_tensor):

        # Split the input into two groups along the last dimension
        def split(input_tensor):
            return tf.split(input_tensor, num_or_size_splits=2, axis=-1)

        split_layer = Lambda(split)(input_tensor)
        
        # First group
        conv1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(split_layer[0])
        dw_conv1 = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
        conv2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(dw_conv1)
        
        # Second group (no modification)
        second_group = split_layer[1]
        
        # Merge the outputs from both groups
        output_tensor = Concatenate()([conv1, conv2, second_group[0], second_group[1]])
        
        return output_tensor
        
    block1_output = block1(conv)
    batch_norm = BatchNormalization()(block1_output)
    flatten_layer = Flatten()(batch_norm)

    def block2(input_tensor):
        # Get the shape of the input tensor
        shape = K.int_shape(input_tensor)
        
        # Reshape the input into four groups
        reshaped_tensor = Reshape((shape[1], shape[2], shape[3]//2, 2))(input_tensor)
        
        # Swap the third and fourth dimensions
        permutated_tensor = Permute((1, 2, 4, 3))(reshaped_tensor)
        
        # Reshape the input back to its original shape
        output_tensor = Reshape((shape[1], shape[2], shape[3]))(permutated_tensor)
        
        return output_tensor
        
    block2_output = block2(flatten_layer)
    final_output = block2_output
    
    dense_layer = Dense(units=10, activation='softmax')(final_output)

    model = keras.Model(inputs=input_layer, outputs=dense_layer)

    return model