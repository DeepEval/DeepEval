import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # First convolutional layer followed by a pooling layer
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    pool1 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(conv1)
    
    # Multi-branch convolutional architecture
    def multi_branch_block(input_tensor):
        # Branch 1: 3x3 convolution
        branch1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        
        # Branch 2: 1x1 convolution followed by two 3x3 convolutions
        branch2 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        branch2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch2)
        branch2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch2)
        
        # Branch 3: Max pooling
        branch3 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(input_tensor)
        
        # Concatenate the outputs from the branches
        output_tensor = Concatenate()([branch1, branch2, branch3])
        
        return output_tensor
    
    # Apply the multi-branch block
    multi_branch_output = multi_branch_block(input_tensor=pool1)
    
    # Batch normalization
    batch_norm = BatchNormalization()(multi_branch_output)
    
    # Flatten the result
    flatten_layer = Flatten()(batch_norm)
    
    # Two fully connected layers for classification
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)
    
    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model