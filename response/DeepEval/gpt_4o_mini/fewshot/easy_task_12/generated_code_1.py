import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Add, Flatten, Dense, Activation
from keras.layers import SeparableConv2D

def dl_model():
    # Input layer
    input_layer = Input(shape=(28, 28, 1))

    # Main Path - Block 1
    x = SeparableConv2D(filters=32, kernel_size=(3, 3), padding='same')(input_layer)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # Main Path - Block 2
    x = SeparableConv2D(filters=64, kernel_size=(3, 3), padding='same')(x)
    x = Activation('relu')(x)
    main_path_output = MaxPooling2D(pool_size=(2, 2))(x)

    # Branch Path
    branch_path = Conv2D(filters=64, kernel_size=(1, 1), padding='same')(input_layer)

    # Summing the outputs from both paths
    combined_output = Add()([main_path_output, branch_path])

    # Flattening and Fully Connected layer
    flatten_layer = Flatten()(combined_output)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    # Constructing the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model