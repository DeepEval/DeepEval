import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Lambda, AveragePooling2D
from tensorflow.keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 with 3 color channels

    # Block 1
    def block1(x):
        # Split the input into three groups
        splits = Lambda(lambda y: tf.split(y, num_or_size_splits=3, axis=3))(x)
        
        # Process each group with a different kernel size
        conv1x1 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(splits[0])
        conv3x3 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(splits[1])
        conv5x5 = Conv2D(filters=32, kernel_size=(5, 5), activation='relu')(splits[2])
        
        # Batch normalization
        conv1x1 = BatchNormalization()(conv1x1)
        conv3x3 = BatchNormalization()(conv3x3)
        conv5x5 = BatchNormalization()(conv5x5)
        
        # Concatenate the outputs
        combined = Concatenate(axis=3)([conv1x1, conv3x3, conv5x5])
        return combined

    block1_output = block1(input_layer)

    # Block 2
    def block2(x):
        # Path 1: 1x1 convolution
        path1 = Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(x)
        
        # Path 2: 3x3 average pooling followed by 1x1 convolution
        path2 = Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(AveragePooling2D(pool_size=(3, 3), strides=1)(x))
        
        # Path 3: 1x1 convolution followed by two sub-paths
        path3 = Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(x)
        sub_path3_1 = Conv2D(filters=32, kernel_size=(1, 3), activation='relu')(path3)
        sub_path3_2 = Conv2D(filters=32, kernel_size=(3, 1), activation='relu')(path3)
        combined_path3 = Concatenate(axis=3)([sub_path3_1, sub_path3_2])
        
        # Path 4: 1x1 convolution followed by two sub-paths
        path4 = Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(x)
        sub_path4_1 = Conv2D(filters=32, kernel_size=(1, 3), activation='relu')(path4)
        sub_path4_2 = Conv2D(filters=32, kernel_size=(3, 1), activation='relu')(path4)
        combined_path4 = Concatenate(axis=3)([sub_path4_1, sub_path4_2])
        
        # Concatenate the outputs of the four paths
        combined_block2 = Concatenate(axis=3)([path1, path2, combined_path3, combined_path4])
        return combined_block2

    block2_output = block2(block1_output)

    # Flatten and fully connected layer
    flatten_layer = Flatten()(block2_output)
    dense_layer = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense_layer)

    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model