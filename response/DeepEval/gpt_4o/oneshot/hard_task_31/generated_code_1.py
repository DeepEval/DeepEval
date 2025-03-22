import keras
from keras.layers import Input, Conv2D, SeparableConv2D, Dropout, Add, Concatenate, Lambda, Flatten, Dense
import tensorflow as tf

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # First Block
    def first_block(input_tensor):
        # Main path
        main_path = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(input_tensor)
        main_path = Dropout(0.25)(main_path)
        main_path = Conv2D(filters=3, kernel_size=(3, 3), padding='same', activation='relu')(main_path)
        
        # Branch path
        branch_path = input_tensor
        
        # Add the main and branch paths
        output_tensor = Add()([main_path, branch_path])
        
        return output_tensor
    
    # Second Block
    def second_block(input_tensor):
        def split_layer(x):
            # Split the input into 3 parts along the channel dimension
            return tf.split(x, num_or_size_splits=3, axis=-1)
        
        # Use Lambda layer to encapsulate the split
        split_tensors = Lambda(split_layer)(input_tensor)
        
        # Define separable convolution paths
        path1 = SeparableConv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(split_tensors[0])
        path1 = Dropout(0.25)(path1)
        
        path2 = SeparableConv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(split_tensors[1])
        path2 = Dropout(0.25)(path2)
        
        path3 = SeparableConv2D(filters=32, kernel_size=(5, 5), padding='same', activation='relu')(split_tensors[2])
        path3 = Dropout(0.25)(path3)
        
        # Concatenate the outputs
        output_tensor = Concatenate()([path1, path2, path3])
        
        return output_tensor

    # Applying the blocks
    block1_output = first_block(input_layer)
    block2_output = second_block(block1_output)

    # Final classification layers
    flatten_layer = Flatten()(block2_output)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model