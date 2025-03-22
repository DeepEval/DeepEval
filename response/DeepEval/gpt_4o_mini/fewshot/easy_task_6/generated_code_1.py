import keras
from keras.layers import Input, Conv2D, Add, Flatten, Dense

def dl_model():
    # Define the input layer
    input_layer = Input(shape=(28, 28, 1))

    # Main path
    # First convolutional layer increases the feature map width
    conv_main_1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    # Second convolutional layer restores the number of channels
    conv_main_2 = Conv2D(filters=1, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv_main_1)

    # Branch path directly connects to the input
    branch_path = input_layer

    # Combine the two paths through an addition operation
    combined_path = Add()([conv_main_2, branch_path])

    # Flatten the combined output and pass it through a fully connected layer
    flatten_layer = Flatten()(combined_path)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    # Create the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model