import keras
from keras.layers import Input, Conv2D, Add, Flatten, Dense

def dl_model():
    # Step 1: Add input layer
    input_layer = Input(shape=(28, 28, 1))

    # Main Path
    # Step 2: Add the first convolutional layer to increase feature map width
    main_conv1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Step 3: Add the second convolutional layer to restore the number of channels
    main_conv2 = Conv2D(filters=1, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(main_conv1)

    # Branch Path
    # Connect directly from the input
    branch = input_layer

    # Step 4: Combine the main path and branch path using addition
    combined = Add()([main_conv2, branch])

    # Step 5: Add a flatten layer
    flatten_layer = Flatten()(combined)

    # Step 6: Add a fully connected dense layer for classification
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    # Step 7: Build the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model