import keras
import tensorflow as tf
from keras.layers import Input, Lambda, SeparableConv2D, BatchNormalization, Concatenate, AveragePooling2D, Conv2D, Flatten, Dense

def dl_model():
    
    input_layer = Input(shape=(32, 32, 3))
    
    def block_1(input_tensor):
        # Split the input into 3 groups
        inputs_groups = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_tensor)
        
        # Apply separable convolutions with different kernel sizes
        conv1 = SeparableConv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(inputs_groups[0])
        conv1 = BatchNormalization()(conv1)
        
        conv2 = SeparableConv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(inputs_groups[1])
        conv2 = BatchNormalization()(conv2)
        
        conv3 = SeparableConv2D(filters=32, kernel_size=(5, 5), padding='same', activation='relu')(inputs_groups[2])
        conv3 = BatchNormalization()(conv3)
        
        # Concatenate the outputs
        output_tensor = Concatenate()([conv1, conv2, conv3])
        return output_tensor

    def block_2(input_tensor):
        # Path 1: 1x1 convolution
        path1 = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(input_tensor)
        
        # Path 2: 3x3 average pooling followed by 1x1 convolution
        path2 = AveragePooling2D(pool_size=(3, 3), strides=1, padding='same')(input_tensor)
        path2 = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(path2)
        
        # Path 3: 1x1 convolution, then split and use 1x3 and 3x1 convolutions
        path3_initial = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(input_tensor)
        path3a = Conv2D(filters=64, kernel_size=(1, 3), padding='same', activation='relu')(path3_initial)
        path3b = Conv2D(filters=64, kernel_size=(3, 1), padding='same', activation='relu')(path3_initial)
        path3 = Concatenate()([path3a, path3b])
        
        # Path 4: 1x1 convolution, 3x3 convolution, then split and use 1x3 and 3x1 convolutions
        path4_initial = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(input_tensor)
        path4_mid = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(path4_initial)
        path4a = Conv2D(filters=64, kernel_size=(1, 3), padding='same', activation='relu')(path4_mid)
        path4b = Conv2D(filters=64, kernel_size=(3, 1), padding='same', activation='relu')(path4_mid)
        path4 = Concatenate()([path4a, path4b])
        
        # Concatenate all paths
        output_tensor = Concatenate()([path1, path2, path3, path4])
        return output_tensor

    # Build the model
    block1_output = block_1(input_layer)
    block2_output = block_2(block1_output)
    
    # Final classification output
    flatten = Flatten()(block2_output)
    output_layer = Dense(units=10, activation='softmax')(flatten)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    return model