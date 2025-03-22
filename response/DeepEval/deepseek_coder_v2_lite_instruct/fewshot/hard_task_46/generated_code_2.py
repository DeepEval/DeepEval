import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, Lambda, Flatten, Dense, GlobalAveragePooling2D
import tensorflow as tf

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 with 3 color channels

    def block_1(input_tensor):
        # Split the input into three groups along the channel axis
        split_groups = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_tensor)
        
        # Apply separable convolutions to each group with different kernel sizes
        conv1x1 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(split_groups[0])
        conv3x3 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(split_groups[1])
        conv5x5 = Conv2D(filters=32, kernel_size=(5, 5), activation='relu')(split_groups[2])
        
        # Concatenate the outputs of the three groups
        output_tensor = Concatenate()([conv1x1, conv3x3, conv5x5])
        return output_tensor

    def block_2(input_tensor):
        # First branch: 3x3 convolution
        conv_branch1 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(input_tensor)
        
        # Second branch: 1x1 convolution followed by two 3x3 convolutions
        conv1x1 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(input_tensor)
        conv3x3_1 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(conv1x1)
        conv3x3_2 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(conv3x3_1)
        
        # Third branch: max pooling
        max_pooling = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(input_tensor)
        
        # Concatenate the outputs of all branches
        output_tensor = Concatenate()([conv_branch1, conv3x3_2, max_pooling])
        return output_tensor

    # Process the input through both blocks
    block1_output = block_1(input_tensor=input_layer)
    block2_output = block_2(input_tensor=block1_output)
    
    # Apply global average pooling and fully connected layer for classification
    global_avg_pool = GlobalAveragePooling2D()(block2_output)
    output_layer = Dense(units=10, activation='softmax')(global_avg_pool)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model