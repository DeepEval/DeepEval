import keras
from keras.layers import Input, DepthwiseConv2D, Conv2D, Dropout, Concatenate, Flatten, Dense
from keras.models import Model

def dl_model():
    # Input layer
    input_layer = Input(shape=(28, 28, 1))

    # Define the specialized block
    def block(input_tensor):
        # Depthwise separable convolution
        depthwise_conv = DepthwiseConv2D(kernel_size=(3, 3), padding='same', activation='relu')(input_tensor)
        conv1x1 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(depthwise_conv)
        dropout1 = Dropout(0.5)(conv1x1)  # Dropout after the 1x1 convolution
        return dropout1

    # Create three branches
    branch1 = block(input_layer)
    branch2 = block(input_layer)
    branch3 = block(input_layer)

    # Concatenate outputs from the three branches
    concatenated = Concatenate()([branch1, branch2, branch3])

    # Flatten the concatenated outputs
    flatten_layer = Flatten()(concatenated)

    # Fully connected layers
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dropout2 = Dropout(0.5)(dense1)  # Dropout to mitigate overfitting
    dense2 = Dense(units=10, activation='softmax')(dropout2)  # Output layer

    # Construct the model
    model = Model(inputs=input_layer, outputs=dense2)

    return model