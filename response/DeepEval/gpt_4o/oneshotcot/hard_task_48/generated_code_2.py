import keras
from keras.layers import Input, SeparableConv2D, BatchNormalization, Concatenate, Lambda, AveragePooling2D, Conv2D, Flatten, Dense
import tensorflow as tf

def dl_model():
    # Step 1: Define input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Step 2: Block 1 - Splitting and applying Separable Convolutions
    def block1(input_tensor):
        # Split the input into 3 groups along the channel dimension
        split_groups = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_tensor)
        
        # Apply Separable Convolution with different kernel sizes on each group
        path1 = SeparableConv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(split_groups[0])
        path1 = BatchNormalization()(path1)
        
        path2 = SeparableConv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(split_groups[1])
        path2 = BatchNormalization()(path2)
        
        path3 = SeparableConv2D(filters=32, kernel_size=(5, 5), padding='same', activation='relu')(split_groups[2])
        path3 = BatchNormalization()(path3)
        
        # Concatenate the outputs of all paths
        output_tensor = Concatenate()([path1, path2, path3])
        return output_tensor
    
    block1_output = block1(input_layer)

    # Step 3: Block 2 with four parallel branches
    def block2(input_tensor):
        # Path 1: 1x1 Convolution
        path1 = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(input_tensor)
        
        # Path 2: 3x3 Average Pooling followed by 1x1 Convolution
        path2 = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(input_tensor)
        path2 = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(path2)
        
        # Path 3: 1x1 Convolution then split to 1x3 and 3x1
        path3 = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(input_tensor)
        path3a = Conv2D(filters=64, kernel_size=(1, 3), padding='same', activation='relu')(path3)
        path3b = Conv2D(filters=64, kernel_size=(3, 1), padding='same', activation='relu')(path3)
        path3 = Concatenate()([path3a, path3b])
        
        # Path 4: 1x1 Convolution then 3x3 Convolution, split to 1x3 and 3x1
        path4 = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(input_tensor)
        path4 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(path4)
        path4a = Conv2D(filters=64, kernel_size=(1, 3), padding='same', activation='relu')(path4)
        path4b = Conv2D(filters=64, kernel_size=(3, 1), padding='same', activation='relu')(path4)
        path4 = Concatenate()([path4a, path4b])
        
        # Concatenate all paths
        output_tensor = Concatenate()([path1, path2, path3, path4])
        return output_tensor
    
    block2_output = block2(block1_output)

    # Step 4: Flatten and Dense for final classification
    flatten_layer = Flatten()(block2_output)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    # Step 5: Create model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model