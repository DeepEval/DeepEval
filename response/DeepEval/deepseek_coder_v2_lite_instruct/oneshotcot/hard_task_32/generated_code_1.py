import keras
from keras.layers import Input, Conv2D, DepthwiseConv2D, SeparableConv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Dropout
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(28, 28, 1))
    
    def specialized_block(input_tensor):
        # Depthwise separable convolutional layer
        depthwise_conv = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        
        # 1x1 convolutional layer
        conv_1x1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(depthwise_conv)
        
        # Dropout to mitigate overfitting
        dropout_1 = Dropout(0.25)(conv_1x1)
        
        return dropout_1
    
    # Three branches
    branch1 = specialized_block(input_layer)
    branch2 = specialized_block(input_layer)
    branch3 = specialized_block(input_layer)
    
    # Concatenate the outputs of the three branches
    concatenated_output = Concatenate()([branch1, branch2, branch3])
    
    # Batch normalization
    batch_norm = BatchNormalization()(concatenated_output)
    
    # Flatten the result
    flatten_layer = Flatten()(batch_norm)
    
    # Two fully connected layers
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dropout_2 = Dropout(0.5)(dense1)
    output_layer = Dense(units=10, activation='softmax')(dropout_2)
    
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model