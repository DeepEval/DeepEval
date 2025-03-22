import keras
import tensorflow as tf
from keras.layers import Input, DepthwiseConv2D, Conv2D, MaxPooling2D, Concatenate, Flatten, Dense, Lambda

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    def block_1(input_tensor):
        # Split input into 3 groups along the last dimension
        split_inputs = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_tensor)
        
        conv1 = DepthwiseConv2D(kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(split_inputs[0])
        conv2 = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(split_inputs[1])
        conv3 = DepthwiseConv2D(kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(split_inputs[2])
        
        output_tensor = Concatenate()([conv1, conv2, conv3])
        return output_tensor
    
    def block_2(input_tensor):
        # Branch 1: 1x1 convolution
        branch1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        
        # Branch 2: 1x1 convolution -> 3x3 convolution -> 3x3 convolution
        branch2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        branch2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch2)
        branch2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch2)
        
        # Branch 3: 1x1 convolution -> 3x3 convolution
        branch3 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        branch3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch3)
        
        # Branch 4: max pooling -> 1x1 convolution
        branch4 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(input_tensor)
        branch4 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(branch4)
        
        # Concatenate all branches
        output_tensor = Concatenate()([branch1, branch2, branch3, branch4])
        return output_tensor

    # Apply the first block
    block1_output = block_1(input_tensor=input_layer)

    # Apply the second block
    block2_output = block_2(input_tensor=block1_output)

    # Flatten and fully connected layer for classification
    flatten_layer = Flatten()(block2_output)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model