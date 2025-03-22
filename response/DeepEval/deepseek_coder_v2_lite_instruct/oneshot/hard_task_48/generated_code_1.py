import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Lambda, AveragePooling2D
from tensorflow.keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 with 3 color channels

    # Block 1
    def block1(x):
        # Split the input into three groups
        split_1 = Lambda(lambda tensor: tensor[:, :11, :11, :])(x)
        split_2 = Lambda(lambda tensor: tensor[:, 7:25, 7:25, :])(x)
        split_3 = Lambda(lambda tensor: tensor[:, 13:31, 13:31, :])(x)
        
        # Process each group with different kernel sizes
        conv_1 = Conv2D(64, (1, 1), activation='relu')(split_1)
        conv_2 = Conv2D(64, (3, 3), activation='relu')(split_2)
        conv_3 = Conv2D(64, (5, 5), activation='relu')(split_3)
        
        # Batch normalization
        conv_1 = BatchNormalization()(conv_1)
        conv_2 = BatchNormalization()(conv_2)
        conv_3 = BatchNormalization()(conv_3)
        
        # Concatenate the outputs
        combined = Concatenate()([conv_1, conv_2, conv_3])
        return combined

    block1_output = block1(input_layer)

    # Block 2
    def block2(x):
        # Path 1: 1x1 convolution
        path1 = Conv2D(64, (1, 1), activation='relu')(x)
        
        # Path 2: 3x3 average pooling followed by 1x1 convolution
        path2 = Conv2D(64, (1, 1), activation='relu')(AveragePooling2D((3, 3))(x))
        
        # Path 3: 1x1 convolution followed by 1x3 and 3x1 convolutions
        path3_1 = Conv2D(64, (1, 1), activation='relu')(x)
        path3_2 = Conv2D(64, (3, 1), activation='relu')(path3_1)
        path3_3 = Conv2D(64, (1, 3), activation='relu')(path3_1)
        path3_combined = Concatenate()([path3_2, path3_3])
        
        # Path 4: 1x1 convolution followed by 3x3 convolution, then 1x3 and 3x1 convolutions
        path4_1 = Conv2D(64, (1, 1), activation='relu')(x)
        path4_2 = Conv2D(64, (3, 3), activation='relu')(path4_1)
        path4_3 = Conv2D(64, (1, 3), activation='relu')(path4_2)
        path4_4 = Conv2D(64, (3, 1), activation='relu')(path4_2)
        path4_combined = Concatenate()([path4_3, path4_4])
        
        # Concatenate the outputs of the four paths
        combined_block2 = Concatenate()([path1, path2, path3_combined, path4_combined])
        return combined_block2

    block2_output = block2(block1_output)

    # Flatten the result and add a fully connected layer for classification
    flatten_layer = Flatten()(block2_output)
    dense_layer = Dense(100, activation='relu')(flatten_layer)
    output_layer = Dense(10, activation='softmax')(dense_layer)

    model = Model(inputs=input_layer, outputs=output_layer)
    return model

# Example usage
model = dl_model()
model.summary()