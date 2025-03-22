import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Dropout
from keras.layers import DepthwiseConv2D

def dl_model():
    input_layer = Input(shape=(28, 28, 1))
    
    def specialized_block(input_tensor):
        # Depthwise Separable Convolution
        depthwise_conv = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        # 1x1 Convolution
        conv1x1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(depthwise_conv)
        # Dropout to mitigate overfitting
        dropout1 = Dropout(0.25)(conv1x1)
        dropout2 = Dropout(0.25)(depthwise_conv)
        return dropout1, dropout2
    
    # Three branches
    branch1_output, branch2_output = specialized_block(input_tensor=input_layer)
    branch3_output = input_layer  # No changes for the first branch
    
    # Concatenate the outputs of the three branches
    concatenated_output = Concatenate()([branch1_output, branch2_output, branch3_output])
    
    # Batch Normalization
    batch_norm = BatchNormalization()(concatenated_output)
    
    # Flatten the result
    flatten_layer = Flatten()(batch_norm)
    
    # Two fully connected layers
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dropout_fc = Dropout(0.4)(dense1)
    dense2 = Dense(units=64, activation='relu')(dropout_fc)
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model