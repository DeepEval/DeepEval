import keras
from keras.layers import Input, Conv2D, AveragePooling2D, BatchNormalization, Concatenate, Flatten, Dense, Lambda
import tensorflow as tf

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 with 3 color channels

    def block_1(input_tensor):
        # Split the input into three groups
        split_groups = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_tensor)
        
        # Process each group with different kernel sizes
        conv1x1 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(split_groups[0])
        conv3x3 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(split_groups[1])
        conv5x5 = Conv2D(filters=32, kernel_size=(5, 5), padding='same', activation='relu')(split_groups[2])
        
        # Batch normalization
        conv1x1 = BatchNormalization()(conv1x1)
        conv3x3 = BatchNormalization()(conv3x3)
        conv5x5 = BatchNormalization()(conv5x5)
        
        # Concatenate the outputs
        concatenated = Concatenate()([conv1x1, conv3x3, conv5x5])
        return concatenated

    def block_2(input_tensor):
        # Path 1: 1x1 convolution
        path1 = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(input_tensor)
        
        # Path 2: 3x3 average pooling followed by 1x1 convolution
        pool_path2 = AveragePooling2D(pool_size=(3, 3), strides=1, padding='same')(input_tensor)
        path2 = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(pool_path2)
        
        # Path 3: 1x1 convolution followed by two sub-paths
        conv_path3 = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(input_tensor)
        sub_path1 = Conv2D(filters=64, kernel_size=(1, 3), padding='same', activation='relu')(conv_path3)
        sub_path2 = Conv2D(filters=64, kernel_size=(3, 1), padding='same', activation='relu')(conv_path3)
        path3 = Concatenate()([sub_path1, sub_path2])
        
        # Path 4: 1x1 convolution followed by 3x3 convolution and two sub-paths
        conv_path4 = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(input_tensor)
        conv_path4_3x3 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(conv_path4)
        sub_path1 = Conv2D(filters=64, kernel_size=(1, 3), padding='same', activation='relu')(conv_path4_3x3)
        sub_path2 = Conv2D(filters=64, kernel_size=(3, 1), padding='same', activation='relu')(conv_path4_3x3)
        path4 = Concatenate()([sub_path1, sub_path2])
        
        # Concatenate outputs of all paths
        concatenated = Concatenate()([path1, path2, path3, path4])
        return concatenated

    # Apply Block 1
    block1_output = block_1(input_tensor=input_layer)
    
    # Apply Block 2
    block2_output = block_2(input_tensor=block1_output)
    
    # Flatten and fully connected layer
    flatten_layer = Flatten()(block2_output)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)
    
    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model