import keras
from keras.layers import Input, Conv2D, Add, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    # Main path
    main_conv1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    main_conv2 = Conv2D(filters=1, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(main_conv1)

    # Branch path (direct connection to input)
    branch = input_layer

    # Combine main and branch paths
    combined = Add()([main_conv2, branch])

    # Flatten and fully connected layer
    flatten_layer = Flatten()(combined)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    # Create model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model