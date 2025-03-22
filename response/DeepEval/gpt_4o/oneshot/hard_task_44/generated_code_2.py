import tensorflow as tf
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, Flatten, Dense, Dropout, Lambda

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # Block 1
    def block1(input_tensor):
        # Split the input into 3 groups along the channel dimension
        split_inputs = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_tensor)
        
        # Apply convolutions with different kernel sizes
        path1 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(split_inputs[0])
        path2 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(split_inputs[1])
        path3 = Conv2D(filters=32, kernel_size=(5, 5), padding='same', activation='relu')(split_inputs[2])
        
        # Apply dropout to each path
        path1 = Dropout(0.5)(path1)
        path2 = Dropout(0.5)(path2)
        path3 = Dropout(0.5)(path3)
        
        # Concatenate the outputs
        output_tensor = Concatenate()([path1, path2, path3])
        
        return output_tensor
    
    # Block 2
    def block2(input_tensor):
        # Branch 1
        branch1 = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(input_tensor)
        
        # Branch 2
        branch2 = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(input_tensor)
        branch2 = Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu')(branch2)
        
        # Branch 3
        branch3 = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(input_tensor)
        branch3 = Conv2D(filters=128, kernel_size=(5, 5), padding='same', activation='relu')(branch3)
        
        # Branch 4
        branch4 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(input_tensor)
        branch4 = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(branch4)
        
        # Concatenate the outputs from all branches
        output_tensor = Concatenate()([branch1, branch2, branch3, branch4])
        
        return output_tensor

    # Construct the model using the blocks
    block1_output = block1(input_layer)
    block2_output = block2(block1_output)
    
    # Flatten the output and apply a fully connected layer
    flatten_layer = Flatten()(block2_output)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)
    
    model = tf.keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model