import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, Flatten, Dense, Lambda, Dropout
import tensorflow as tf

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # Block 1: Split into three groups and apply different convolutions
    def block1(input_tensor):
        # Split input into three groups along the channel dimension
        split_channels = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_tensor)
        
        # Apply different convolutions to each group
        conv1 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(split_channels[0])
        conv2 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(split_channels[1])
        conv3 = Conv2D(filters=32, kernel_size=(5, 5), padding='same', activation='relu')(split_channels[2])
        
        # Concatenate the outputs from the three groups
        concat = Concatenate()([conv1, conv2, conv3])
        
        # Apply dropout to reduce overfitting
        dropout = Dropout(rate=0.5)(concat)
        
        return dropout

    block1_output = block1(input_layer)

    # Block 2: Four branches with different operations
    def block2(input_tensor):
        # Branch 1: 1x1 convolution
        branch1 = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(input_tensor)

        # Branch 2: 1x1 convolution followed by 3x3 convolution
        branch2 = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(input_tensor)
        branch2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(branch2)

        # Branch 3: 1x1 convolution followed by 5x5 convolution
        branch3 = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(input_tensor)
        branch3 = Conv2D(filters=64, kernel_size=(5, 5), padding='same', activation='relu')(branch3)

        # Branch 4: 3x3 max pooling followed by 1x1 convolution
        branch4 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(input_tensor)
        branch4 = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(branch4)

        # Concatenate outputs of all branches
        output_tensor = Concatenate()([branch1, branch2, branch3, branch4])
        
        return output_tensor

    block2_output = block2(block1_output)

    # Flatten and fully connected layer
    flatten_layer = Flatten()(block2_output)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    # Build model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model