import keras
from keras.layers import Input, AveragePooling2D, Flatten, Concatenate, Dense, Dropout, Reshape
from keras.models import Model
import tensorflow as tf

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    def block_1(input_tensor):
        # Three parallel paths with different average pooling layers
        pool1 = AveragePooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(input_tensor)
        flat1 = Flatten()(pool1)
        pool2 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(input_tensor)
        flat2 = Flatten()(pool2)
        pool3 = AveragePooling2D(pool_size=(4, 4), strides=(4, 4), padding='same')(input_tensor)
        flat3 = Flatten()(pool3)
        
        # Concatenate the flattened outputs
        concat = Concatenate()([flat1, flat2, flat3])
        
        # Apply dropout for regularization
        dropout = Dropout(0.5)(concat)
        
        return dropout

    def block_2(input_tensor):
        # Four branches for feature extraction
        branch1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        branch2 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        branch3 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        branch4 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        branch5 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        branch6 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        
        # Concatenate the outputs of the branches
        concat = Concatenate()([branch1, branch2, branch3, branch4, branch5, branch6])
        
        return concat

    # Process Block 1
    block1_output = block_1(input_layer)
    
    # Reshape the output of Block 1 to a 4D tensor
    reshaped = Reshape(target_shape=(4, 4, 4))(block1_output)
    
    # Process Block 2
    block2_output = block_2(reshaped)
    
    # Flatten the output of Block 2
    flatten = Flatten()(block2_output)
    
    # Output layer with two fully connected layers
    output_layer = Dense(units=10, activation='softmax')(flatten)
    
    # Construct the model
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model