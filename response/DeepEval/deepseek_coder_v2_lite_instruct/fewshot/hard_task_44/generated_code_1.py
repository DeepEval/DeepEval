import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, Dropout, Flatten, Dense, Lambda
import tensorflow as tf

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    def block_1(input_tensor):
        # Split the input into three groups
        split_groups = Lambda(lambda x: tf.split(x, 3, axis=-1))(input_tensor)
        
        # Feature extraction for each group
        conv1 = Conv2D(32, (1, 1), activation='relu')(split_groups[0])
        conv2 = Conv2D(32, (3, 3), activation='relu')(split_groups[1])
        conv3 = Conv2D(32, (5, 5), activation='relu')(split_groups[2])
        
        # Concatenate the outputs
        concatenated = Concatenate()([conv1, conv2, conv3])
        
        # Apply dropout for regularization
        dropout_layer = Dropout(0.5)(concatenated)
        
        return dropout_layer

    def block_2(input_tensor):
        # Branch 1: 1x1 convolution
        branch1 = Conv2D(32, (1, 1), activation='relu')(input_tensor)
        
        # Branch 2: <1x1 convolution, 3x3 convolution>
        branch2a = Conv2D(32, (1, 1), activation='relu')(input_tensor)
        branch2b = Conv2D(32, (3, 3), activation='relu')(branch2a)
        
        # Branch 3: <1x1 convolution, 5x5 convolution>
        branch3a = Conv2D(32, (1, 1), activation='relu')(input_tensor)
        branch3b = Conv2D(32, (5, 5), activation='relu')(branch3a)
        
        # Branch 4: <3x3 max pooling, 1x1 convolution>
        branch4a = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(input_tensor)
        branch4b = Conv2D(32, (1, 1), activation='relu')(branch4a)
        
        # Concatenate all branches
        concatenated = Concatenate()([branch1, branch2b, branch3b, branch4b])
        
        return concatenated

    # Applying the blocks
    block1_output = block_1(input_tensor=input_layer)
    block2_output = block_2(input_tensor=block1_output)

    # Flatten the output and add fully connected layers for classification
    flatten_layer = Flatten()(block2_output)
    dense_layer = Dense(10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=dense_layer)

    return model