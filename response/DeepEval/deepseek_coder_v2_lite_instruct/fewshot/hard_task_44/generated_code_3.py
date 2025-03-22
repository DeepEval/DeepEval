import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, Flatten, Dense, Dropout, Lambda
import tensorflow as tf

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    def block_1(input_tensor):
        # Split the input into three groups along the channel axis
        split_groups = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_tensor)
        
        # Apply convolutions to each group
        conv1x1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(split_groups[0])
        conv3x3 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(split_groups[1])
        conv5x5 = Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(split_groups[2])
        
        # Concatenate the outputs
        concatenated = Concatenate()([conv1x1, conv3x3, conv5x5])
        
        # Apply dropout to reduce overfitting
        dropout_output = Dropout(0.5)(concatenated)
        
        return dropout_output

    def block_2(input_tensor):
        # Define the four branches
        branch1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        branch2 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        branch3 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch2)
        branch4 = Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(branch2)
        branch5 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(input_tensor)
        branch6 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(branch5)
        
        # Concatenate the outputs from all branches
        concatenated = Concatenate()([branch1, branch3, branch4, branch6])
        
        return concatenated

    # Apply block 1 to the input
    block1_output = block_1(input_tensor=input_layer)
    
    # Apply block 2 to the output of block 1
    block2_output = block_2(input_tensor=block1_output)
    
    # Flatten the output from block 2
    flattened = Flatten()(block2_output)
    
    # Pass the flattened output through a fully connected layer
    output_layer = Dense(units=10, activation='softmax')(flattened)

    # Construct and return the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model