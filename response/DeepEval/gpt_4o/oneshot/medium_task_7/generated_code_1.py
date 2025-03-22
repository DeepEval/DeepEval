import keras
from keras.layers import Input, Conv2D, Add, Dense, Flatten

def dl_model():
    # Define the input layer
    input_layer = Input(shape=(32, 32, 3))

    # First sequential convolutional layer
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(input_layer)

    # Second sequential convolutional layer
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(conv1)

    # Third sequential convolutional layer
    conv3 = Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu')(conv2)

    # Add outputs of the first two convolutional layers with the output of the third
    added_outputs = Add()([conv1, conv2, conv3])

    # A separate convolutional layer processing the input directly
    direct_conv = Conv2D(filters=128, kernel_size=(1, 1), padding='same', activation='relu')(input_layer)

    # Combine the added outputs and the direct path
    combined_output = Add()([added_outputs, direct_conv])

    # Flatten the output
    flatten_layer = Flatten()(combined_output)

    # First fully connected layer
    dense1 = Dense(units=256, activation='relu')(flatten_layer)

    # Second fully connected layer for classification
    output_layer = Dense(units=10, activation='softmax')(dense1)

    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model