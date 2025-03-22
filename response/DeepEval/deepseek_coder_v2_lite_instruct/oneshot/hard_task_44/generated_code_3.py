import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, Lambda, Dropout, Flatten, Dense
from tensorflow.keras.models import Model

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Block 1
    def block1(x):
        # Split the input into three groups
        split1 = Lambda(lambda tensor: tensor[:, :, :, :tensor.shape[3]//3])(x)
        split2 = Lambda(lambda tensor: tensor[:, :, :, tensor.shape[3]//3:2*tensor.shape[3]//3])(x)
        split3 = Lambda(lambda tensor: tensor[:, :, :, 2*tensor.shape[3]//3:])(x)
        
        # Apply convolutions to each group
        conv1x1 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(split1)
        conv3x3 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(split2)
        conv5x5 = Conv2D(filters=32, kernel_size=(5, 5), activation='relu')(split3)
        
        # Concatenate the outputs
        concatenated = Concatenate()([conv1x1, conv3x3, conv5x5])
        
        # Apply dropout
        dropout_output = Dropout(0.5)(concatenated)
        
        return dropout_output
    
    block1_output = block1(input_layer)
    
    # Block 2
    def block2(x):
        # Branch 1: 1x1 convolution
        branch1 = Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(x)
        
        # Branch 2: <1x1 convolution, 3x3 convolution>
        branch2_1 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(x)
        branch2_2 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(branch2_1)
        
        # Branch 3: <1x1 convolution, 5x5 convolution>
        branch3_1 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(x)
        branch3_2 = Conv2D(filters=32, kernel_size=(5, 5), activation='relu')(branch3_1)
        
        # Branch 4: <3x3 max pooling, 1x1 convolution>
        branch4_1 = MaxPooling2D(pool_size=(3, 3), strides=1, padding='same')(x)
        branch4_2 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(branch4_1)
        
        # Concatenate the outputs
        concatenated = Concatenate()([branch1, branch2_2, branch3_2, branch4_2])
        
        return concatenated
    
    block2_output = block2(block1_output)
    
    # Flatten the output
    flattened = Flatten()(block2_output)
    
    # Fully connected layer
    output_layer = Dense(units=10, activation='softmax')(flattened)
    
    # Create the model
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model