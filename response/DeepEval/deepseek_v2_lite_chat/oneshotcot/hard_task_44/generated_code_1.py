import keras
from keras.layers import Input, Lambda, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Activation

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Split input into three groups along the channel
    split_layer = Lambda(lambda x: tf.split(x, 3, axis=-1))(input_layer)
    
    # Block 1: Feature extraction with varying kernel sizes
    def block1(input_tensor):
        # Convolutional layers
        conv1 = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(input_tensor[0])
        conv2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(input_tensor[1])
        conv3 = Conv2D(filters=64, kernel_size=(5, 5), padding='same', activation='relu')(input_tensor[2])
        
        # Dropout layer
        dropout = Dropout(rate=0.2)(conv3)
        
        # Concatenate outputs
        concat = Concatenate(axis=-1)([conv1, conv2, dropout])
        
        return concat
    
    block1_output = block1(split_layer)
    
    # Block 2: Separate processing with different branches
    def block2(input_tensor):
        # Branches
        branch1 = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(input_tensor)
        branch2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(input_tensor)
        branch3 = Conv2D(filters=64, kernel_size=(5, 5), padding='same', activation='relu')(input_tensor)
        branch4 = MaxPooling2D(pool_size=(3, 3), padding='same')(input_tensor)
        
        # Concatenate all branches
        concat_branch = Concatenate(axis=-1)([branch1, branch2, branch3, branch4])
        
        return concat_branch
    
    block2_output = block2(block1_output)
    
    # Flatten and fully connected layers
    flatten = Flatten()(block2_output)
    dense = Dense(units=512, activation='relu')(flatten)
    output_layer = Dense(units=10, activation='softmax')(dense)
    
    # Model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model