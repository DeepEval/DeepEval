import keras
from keras.layers import Input, SeparableConv2D, Conv2D, Dropout, Concatenate, Flatten, Dense
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(28, 28, 1))
    
    def branch(input_tensor):
        # Depthwise separable convolution
        depthwise_conv = SeparableConv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(input_tensor)
        dropout1 = Dropout(0.3)(depthwise_conv)
        
        # 1x1 Convolution
        pointwise_conv = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(dropout1)
        dropout2 = Dropout(0.3)(pointwise_conv)

        return dropout2

    # Create three branches
    branch1 = branch(input_layer)
    branch2 = branch(input_layer)
    branch3 = branch(input_layer)

    # Concatenate the outputs of the branches
    concatenated = Concatenate()([branch1, branch2, branch3])

    # Flatten the concatenated outputs
    flatten_layer = Flatten()(concatenated)

    # Fully connected layers
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    # Define the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model