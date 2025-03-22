import keras
from keras.layers import Input, Lambda, SeparableConv2D, BatchNormalization, Concatenate, Conv2D, AveragePooling2D, Flatten, Dense
import tensorflow as tf

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # First Block
    def first_block(input_tensor):
        def split_and_process(input_tensor):
            splits = tf.split(input_tensor, num_or_size_splits=3, axis=-1)
            path1 = SeparableConv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(splits[0])
            path2 = SeparableConv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(splits[1])
            path3 = SeparableConv2D(filters=32, kernel_size=(5, 5), padding='same', activation='relu')(splits[2])
            return [path1, path2, path3]

        processed_splits = Lambda(split_and_process)(input_tensor)
        normalized_splits = [BatchNormalization()(output) for output in processed_splits]
        output_tensor = Concatenate()(normalized_splits)

        return output_tensor
    
    block1_output = first_block(input_tensor=input_layer)
    
    # Second Block
    def second_block(input_tensor):
        # Branch 1
        branch1 = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(input_tensor)
        branch1 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(branch1)
        
        # Branch 2
        branch2 = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(input_tensor)
        branch2 = Conv2D(filters=64, kernel_size=(1, 7), padding='same', activation='relu')(branch2)
        branch2 = Conv2D(filters=64, kernel_size=(7, 1), padding='same', activation='relu')(branch2)
        branch2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(branch2)
        
        # Branch 3
        branch3 = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(input_tensor)
        
        # Concatenate all branches
        output_tensor = Concatenate()([branch1, branch2, branch3])

        return output_tensor
    
    block2_output = second_block(input_tensor=block1_output)
    
    # Flatten and Fully Connected Layers
    flatten_layer = Flatten()(block2_output)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)
    
    # Building the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model