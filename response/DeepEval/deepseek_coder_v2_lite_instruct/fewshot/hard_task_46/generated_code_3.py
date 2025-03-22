import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, SeparableConv2D, Lambda, GlobalAveragePooling2D, Dense, Reshape
import tensorflow as tf

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    def block_1(input_tensor):
        # Split the input into three groups along the channel axis
        groups = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_tensor)
        
        # Apply separable convolutions to each group
        conv1 = SeparableConv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(groups[0])
        conv2 = SeparableConv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(groups[1])
        conv3 = SeparableConv2D(filters=32, kernel_size=(5, 5), padding='same', activation='relu')(groups[2])
        
        # Concatenate the outputs of the three groups
        output_tensor = Concatenate()([conv1, conv2, conv3])
        return output_tensor

    def block_2(input_tensor):
        # First branch: 3x3 convolution
        conv1 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(input_tensor)
        
        # Second branch: 1x1 convolution followed by two 3x3 convolutions
        conv2_1 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(input_tensor)
        conv2_2 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(conv2_1)
        conv2_3 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(conv2_2)
        
        # Third branch: max pooling
        max_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(input_tensor)
        
        # Concatenate the outputs of all branches
        output_tensor = Concatenate()([conv1, conv2_2, max_pool])
        return output_tensor

    # Process input through the first block
    block1_output = block_1(input_tensor=input_layer)
    
    # Process the output of the first block through the second block
    block2_output = block_2(input_tensor=block1_output)
    
    # Apply global average pooling
    gap = GlobalAveragePooling2D()(block2_output)
    
    # Fully connected layer for classification
    output_layer = Dense(units=10, activation='softmax')(gap)

    model = keras.Model(inputs=input_layer, outputs=output_layer)
    return model