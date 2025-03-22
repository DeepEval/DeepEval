import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, Flatten, Dense, Dropout, Lambda
import tensorflow as tf

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 with 3 color channels

    # Block 1: Splitting into three groups, applying convolutions, concatenation, and dropout
    def block_1(input_tensor):
        # Splitting the input into three groups along the channel axis
        groups = Lambda(lambda x: tf.split(x, 3, axis=-1))(input_tensor)
        
        # Applying convolutions to each group
        conv1 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(groups[0])
        conv2 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(groups[1])
        conv3 = Conv2D(filters=32, kernel_size=(5, 5), activation='relu')(groups[2])
        
        # Concatenating the outputs
        concatenated = Concatenate()([conv1, conv2, conv3])
        
        # Adding dropout for regularization
        dropout = Dropout(0.2)(concatenated)
        
        return dropout

    # Block 2: Four branches for processing the input
    def block_2(input_tensor):
        # Branches
        branch1 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(input_tensor)
        branch2 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(input_tensor)
        branch3 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(input_tensor)
        branch4 = Conv2D(filters=32, kernel_size=(5, 5), activation='relu')(input_tensor)
        branch5 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(input_tensor)
        branch6 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(branch5)
        
        # Concatenating the outputs
        concatenated = Concatenate()([branch1, branch2, branch3, branch4, branch6])
        
        return concatenated

    # Applying Block 1
    block1_output = block_1(input_tensor=input_layer)
    
    # Applying Block 2
    block2_output = block_2(input_tensor=block1_output)
    
    # Flattening the output
    flatten = Flatten()(block2_output)
    
    # Final classification layer
    output_layer = Dense(units=10, activation='softmax')(flatten)

    # Constructing the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model