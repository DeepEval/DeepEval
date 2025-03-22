import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, Lambda, Flatten, Dense, DepthwiseConv2D
import tensorflow as tf

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    def block_1(input_tensor):
        # Split the input into three groups along the last dimension
        groups = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_tensor)
        
        # Apply depthwise separable convolutions to each group
        conv1 = DepthwiseConv2D(kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(groups[0])
        conv2 = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(groups[1])
        conv3 = DepthwiseConv2D(kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(groups[2])
        
        # Concatenate the outputs
        output_tensor = Concatenate()([conv1, conv2, conv3])
        return output_tensor

    def block_2(input_tensor):
        # Define multiple branches for feature extraction
        branch1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        branch2 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        branch3 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch2)
        branch4 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch2)
        branch5 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(input_tensor)
        branch6 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(branch5)

        # Concatenate the outputs of all branches
        output_tensor = Concatenate()([branch1, branch3, branch4, branch6])
        return output_tensor

    # Apply both blocks to the input layer
    block1_output = block_1(input_tensor=input_layer)
    block2_output = block_2(input_tensor=block1_output)

    # Flatten the output and add fully connected layers for classification
    flatten_layer = Flatten()(block2_output)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    # Create and return the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    return model