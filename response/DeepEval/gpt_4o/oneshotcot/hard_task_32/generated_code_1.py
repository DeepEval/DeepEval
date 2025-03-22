from keras.layers import Input, SeparableConv2D, Conv2D, Dropout, Concatenate, Flatten, Dense
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(28, 28, 1))
    
    def specialized_block(input_tensor):
        # Depthwise Separable Convolutional Layer
        depthwise_sep_conv = SeparableConv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(input_tensor)
        dropout1 = Dropout(rate=0.3)(depthwise_sep_conv)
        
        # 1x1 Convolutional Layer
        conv_1x1 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(dropout1)
        dropout2 = Dropout(rate=0.3)(conv_1x1)
        
        return dropout2

    # Branch 1
    branch1 = specialized_block(input_layer)

    # Branch 2
    branch2 = specialized_block(input_layer)

    # Branch 3
    branch3 = specialized_block(input_layer)

    # Concatenate Outputs of All Branches
    concatenated = Concatenate()([branch1, branch2, branch3])
    
    # Flatten and Fully Connected Layers
    flatten_layer = Flatten()(concatenated)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    # Construct the Model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model