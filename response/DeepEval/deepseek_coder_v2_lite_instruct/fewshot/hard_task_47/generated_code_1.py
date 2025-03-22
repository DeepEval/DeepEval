import keras
from keras.layers import Input, Lambda, Conv2D, BatchNormalization, Concatenate, Dense, AveragePooling2D, GlobalAveragePooling2D
import tensorflow as tf

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    def block_1(input_tensor):
        # Split the input into three groups
        groups = Lambda(lambda x: tf.split(x, 3, axis=-1))(input_tensor)
        
        # Process each group with depthwise separable convolutional layers
        conv1 = DepthwiseConv2D(kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(groups[0])
        conv2 = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(groups[1])
        conv3 = DepthwiseConv2D(kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(groups[2])
        
        # Apply batch normalization
        conv1 = BatchNormalization()(conv1)
        conv2 = BatchNormalization()(conv2)
        conv3 = BatchNormalization()(conv3)
        
        # Concatenate the outputs
        output_tensor = Concatenate()([conv1, conv2, conv3])
        return output_tensor

    def block_2(input_tensor):
        # Define branches
        branch1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        branch1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch1)
        
        branch2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        branch2 = Conv2D(filters=64, kernel_size=(1, 7), strides=(1, 1), padding='same', activation='relu')(branch2)
        branch2 = Conv2D(filters=64, kernel_size=(7, 1), strides=(1, 1), padding='same', activation='relu')(branch2)
        branch2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch2)
        
        branch3 = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(input_tensor)
        branch3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch3)
        
        # Concatenate the outputs of all branches
        output_tensor = Concatenate()([branch1, branch2, branch3])
        return output_tensor

    # Process the input through the first block
    block1_output = block_1(input_tensor=input_layer)
    
    # Process the output of the first block through the second block
    block2_output = block_2(input_tensor=block1_output)
    
    # Flatten the output and pass it through fully connected layers
    flatten = Flatten()(block2_output)
    dense1 = Dense(units=256, activation='relu')(flatten)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)
    return model