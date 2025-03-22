import keras
import tensorflow as tf
from keras.layers import Input, DepthwiseConv2D, Conv2D, MaxPooling2D, Lambda, Concatenate, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 pixels with 3 color channels

    def block_1(input_tensor):
        # Splitting the input into three groups along the last dimension
        input_groups = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_tensor)
        
        # Applying depthwise separable convolutions with different kernel sizes
        conv1 = DepthwiseConv2D(kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_groups[0])
        conv2 = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_groups[1])
        conv3 = DepthwiseConv2D(kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(input_groups[2])
        
        # Concatenating the outputs
        output_tensor = Concatenate()([conv1, conv2, conv3])
        return output_tensor

    def block_2(input_tensor):
        # Branch 1: 1x1 convolution
        branch1 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(input_tensor)
        
        # Branch 2: 1x1 convolution, 3x3 convolution, 3x3 convolution
        branch2_1 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(input_tensor)
        branch2_2 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(branch2_1)
        branch2 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(branch2_2)
        
        # Branch 3: 1x1 convolution, 3x3 convolution
        branch3_1 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(input_tensor)
        branch3 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(branch3_1)
        
        # Branch 4: max pooling, 1x1 convolution
        branch4_1 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(input_tensor)
        branch4 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(branch4_1)
        
        # Concatenating the outputs from all branches
        output_tensor = Concatenate()([branch1, branch2, branch3, branch4])
        return output_tensor

    # Applying the first block
    block1_output = block_1(input_tensor=input_layer)
    
    # Applying the second block
    block2_output = block_2(input_tensor=block1_output)

    # Flattening and fully connected layer for classification
    flatten_layer = Flatten()(block2_output)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)  # CIFAR-10 has 10 classes

    # Constructing the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model