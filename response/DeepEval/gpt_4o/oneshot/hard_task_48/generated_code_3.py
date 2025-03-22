import keras
from keras.layers import Input, Lambda, SeparableConv2D, BatchNormalization, Concatenate, AveragePooling2D, Conv2D, Flatten, Dense
import tensorflow as tf

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # Block 1
    def block1(input_tensor):
        split1, split2, split3 = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_tensor)
        
        # Separable Convolutional Layers with different kernel sizes
        conv1 = SeparableConv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(split1)
        conv2 = SeparableConv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(split2)
        conv3 = SeparableConv2D(filters=32, kernel_size=(5, 5), padding='same', activation='relu')(split3)
        
        # Batch Normalization
        bn1 = BatchNormalization()(conv1)
        bn2 = BatchNormalization()(conv2)
        bn3 = BatchNormalization()(conv3)
        
        # Concatenate the outputs of the three groups
        output_tensor = Concatenate()([bn1, bn2, bn3])
        
        return output_tensor
    
    block1_output = block1(input_layer)
    
    # Block 2
    def block2(input_tensor):
        # Path 1: 1x1 Convolution
        path1 = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(input_tensor)
        
        # Path 2: 3x3 Average Pooling followed by 1x1 Convolution
        path2 = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(input_tensor)
        path2 = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(path2)
        
        # Path 3: 1x1 Convolution then split into 1x3 and 3x1 Convolutions
        path3 = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(input_tensor)
        path3_1 = Conv2D(filters=64, kernel_size=(1, 3), padding='same', activation='relu')(path3)
        path3_2 = Conv2D(filters=64, kernel_size=(3, 1), padding='same', activation='relu')(path3)
        path3 = Concatenate()([path3_1, path3_2])
        
        # Path 4: 1x1 Convolution, 3x3 Convolution, split into 1x3 and 3x1 Convolutions
        path4 = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(input_tensor)
        path4 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(path4)
        path4_1 = Conv2D(filters=64, kernel_size=(1, 3), padding='same', activation='relu')(path4)
        path4_2 = Conv2D(filters=64, kernel_size=(3, 1), padding='same', activation='relu')(path4)
        path4 = Concatenate()([path4_1, path4_2])
        
        # Concatenate the outputs of the four paths
        output_tensor = Concatenate()([path1, path2, path3, path4])
        
        return output_tensor
    
    block2_output = block2(block1_output)
    
    # Flatten and Fully Connected Layer for final classification
    flatten_layer = Flatten()(block2_output)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model