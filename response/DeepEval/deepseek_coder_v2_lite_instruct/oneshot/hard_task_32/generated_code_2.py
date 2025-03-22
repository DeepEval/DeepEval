import keras
from keras.layers import Input, Conv2D, DepthwiseConv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Dropout

def dl_model():
    input_layer = Input(shape=(28, 28, 1))
    
    def branch(input_tensor):
        # Depthwise separable convolutional layer
        conv_ds = DepthwiseConv2D(kernel_size=(3, 3), padding='same', activation='relu')(input_tensor)
        # 1x1 convolutional layer
        conv_1x1 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(conv_ds)
        # Dropout layer
        dropout = Dropout(0.25)(conv_1x1)
        return dropout
    
    # Create three branches
    branch1 = branch(input_layer)
    branch2 = branch(input_layer)
    branch3 = branch(input_layer)
    
    # Concatenate outputs from the three branches
    concatenated = Concatenate()([branch1, branch2, branch3])
    
    # Batch normalization
    batch_norm = BatchNormalization()(concatenated)
    
    # Flatten the result
    flatten_layer = Flatten()(batch_norm)
    
    # Two fully connected layers
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dropout_fc = Dropout(0.5)(dense1)
    output_layer = Dense(units=10, activation='softmax')(dropout_fc)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model