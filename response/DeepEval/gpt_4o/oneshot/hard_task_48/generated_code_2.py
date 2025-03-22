import keras
from keras.layers import Input, Conv2D, SeparableConv2D, AveragePooling2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Lambda
import tensorflow as tf

def dl_model():
    
    input_layer = Input(shape=(32, 32, 3))
    
    # Block 1
    def block1(input_tensor):
        
        # Split into three groups
        splits = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_tensor)
        
        group1 = SeparableConv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(splits[0])
        group2 = SeparableConv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(splits[1])
        group3 = SeparableConv2D(filters=32, kernel_size=(5, 5), padding='same', activation='relu')(splits[2])
        
        # Batch Normalization
        group1 = BatchNormalization()(group1)
        group2 = BatchNormalization()(group2)
        group3 = BatchNormalization()(group3)
        
        # Concatenate outputs
        output_tensor = Concatenate()([group1, group2, group3])
        
        return output_tensor
    
    block1_output = block1(input_layer)
    
    # Block 2
    def block2(input_tensor):
        
        # Path 1: 1x1 Convolution
        path1 = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(input_tensor)
        
        # Path 2: 3x3 Average Pooling -> 1x1 Convolution
        path2 = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(input_tensor)
        path2 = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(path2)
        
        # Path 3: 1x1 Convolution -> 1x3 and 3x1 Convolutions
        path3 = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(input_tensor)
        path3_1 = Conv2D(filters=64, kernel_size=(1, 3), padding='same', activation='relu')(path3)
        path3_2 = Conv2D(filters=64, kernel_size=(3, 1), padding='same', activation='relu')(path3)
        path3_output = Concatenate()([path3_1, path3_2])
        
        # Path 4: 1x1 Convolution -> 3x3 Convolution -> 1x3 and 3x1 Convolutions
        path4 = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(input_tensor)
        path4 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(path4)
        path4_1 = Conv2D(filters=64, kernel_size=(1, 3), padding='same', activation='relu')(path4)
        path4_2 = Conv2D(filters=64, kernel_size=(3, 1), padding='same', activation='relu')(path4)
        path4_output = Concatenate()([path4_1, path4_2])
        
        # Concatenate all paths
        output_tensor = Concatenate()([path1, path2, path3_output, path4_output])
        
        return output_tensor
    
    block2_output = block2(block1_output)
    
    # Flatten and Fully Connected Layer
    flatten_layer = Flatten()(block2_output)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)
    
    # Define the Model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Create the model
model = dl_model()