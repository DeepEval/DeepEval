import keras
from keras.layers import Input, Conv2D, Add, Flatten, Dense

def dl_model():
    # Define the input layer
    input_layer = Input(shape=(28, 28, 1))

    # Main path
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
    main_path = conv2

    # Branch path
    branch_path = input_layer

    # Addition operation
    added = Add()([main_path, branch_path])

    # Flatten the output
    flattened = Flatten()(added)

    # Fully connected layer
    output_layer = Dense(units=10, activation='softmax')(flattened)

    # Create the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model