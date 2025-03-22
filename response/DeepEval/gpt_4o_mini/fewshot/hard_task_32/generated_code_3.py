import keras
from keras.layers import Input, DepthwiseConv2D, Conv2D, Dropout, Concatenate, Flatten, Dense
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    def branch_block(input_tensor):
        # Depthwise separable convolution
        depthwise_conv = DepthwiseConv2D(kernel_size=(3, 3), padding='same', activation='relu')(input_tensor)
        # 1x1 convolution
        conv_1x1 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(depthwise_conv)
        # Dropout to mitigate overfitting
        dropout = Dropout(0.5)(conv_1x1)
        return dropout

    # Create three branches
    branch1_output = branch_block(input_layer)
    branch2_output = branch_block(input_layer)
    branch3_output = branch_block(input_layer)

    # Concatenate the outputs from the branches
    concatenated = Concatenate()([branch1_output, branch2_output, branch3_output])
    
    # Flatten the concatenated output
    flatten_layer = Flatten()(concatenated)
    
    # Fully connected layers
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dropout_dense = Dropout(0.5)(dense1)  # Dropout to mitigate overfitting
    output_layer = Dense(units=10, activation='softmax')(dropout_dense)

    # Construct the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model