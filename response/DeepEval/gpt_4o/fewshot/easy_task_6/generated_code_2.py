import keras
from keras.layers import Input, Conv2D, Add, Flatten, Dense

def dl_model():
    
    input_layer = Input(shape=(28, 28, 1))

    # Main path
    # First convolutional layer increases the feature map width to 64 channels
    conv_main_1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    # Second convolutional layer restores the number of channels back to 1
    conv_main_2 = Conv2D(filters=1, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv_main_1)

    # Branch path
    # Directly connected to the input
    branch_path = input_layer

    # Combine the outputs of both paths using an addition operation
    combined = Add()([conv_main_2, branch_path])

    # Flatten and connect to fully connected layer for classification
    flatten_layer = Flatten()(combined)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    # Create the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model