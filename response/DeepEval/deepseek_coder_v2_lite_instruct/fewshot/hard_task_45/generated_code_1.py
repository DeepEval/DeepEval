import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, DepthwiseConv2D, Lambda, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    def block_1(input_tensor):
        # Splitting the input into three groups
        groups = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_tensor)
        
        # Depthwise separable convolutions on each group
        conv1 = DepthwiseConv2D(kernel_size=(1, 1), padding='same', activation='relu')(groups[0])
        conv2 = DepthwiseConv2D(kernel_size=(3, 3), padding='same', activation='relu')(groups[1])
        conv3 = DepthwiseConv2D(kernel_size=(5, 5), padding='same', activation='relu')(groups[2])
        
        # Concatenating the outputs of the three groups
        output_tensor = Concatenate()([conv1, conv2, conv3])
        return output_tensor

    def block_2(input_tensor):
        # Branch 1: 1x1 convolution
        branch1 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(input_tensor)
        
        # Branch 2: <1x1 convolution
        branch2 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(input_tensor)
        branch2 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(branch2)
        
        # Branch 3: 3x3 convolution
        branch3 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(input_tensor)
        
        # Branch 4: 3x3 convolution
        branch4 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(input_tensor)
        
        # Branch 5: <1x1 convolution, 3x3 convolution>
        branch5 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(input_tensor)
        branch5 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(branch5)
        
        # Branch 6: <max pooling, 1x1 convolution>
        branch6 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(input_tensor)
        branch6 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(branch6)
        
        # Concatenating all branch outputs
        output_tensor = Concatenate()([branch1, branch2, branch3, branch4, branch5, branch6])
        return output_tensor

    block1_output = block_1(input_tensor=input_layer)
    block2_output = block_2(input_tensor=block1_output)

    flatten_layer = Flatten()(block2_output)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = tf.keras.Model(inputs=input_layer, outputs=output_layer)
    return model