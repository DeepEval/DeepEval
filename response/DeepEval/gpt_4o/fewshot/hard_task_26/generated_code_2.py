import keras
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate, Add, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Initial 1x1 Convolution
    initial_conv = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(input_layer)

    # Branch 1: 3x3 Convolution
    branch1 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(initial_conv)

    # Branch 2: MaxPooling -> 3x3 Convolution -> UpSampling
    branch2_down = MaxPooling2D(pool_size=(2, 2), padding='same')(initial_conv)
    branch2_conv = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(branch2_down)
    branch2_up = UpSampling2D(size=(2, 2))(branch2_conv)

    # Branch 3: MaxPooling -> 3x3 Convolution -> UpSampling
    branch3_down = MaxPooling2D(pool_size=(2, 2), padding='same')(initial_conv)
    branch3_conv = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(branch3_down)
    branch3_up = UpSampling2D(size=(2, 2))(branch3_conv)

    # Concatenate all branches
    concatenated = Concatenate()([branch1, branch2_up, branch3_up])

    # Final 1x1 Convolution for main path
    main_output = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(concatenated)

    # Branch path: 1x1 Convolution
    branch_path = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(input_layer)

    # Add outputs from main path and branch path
    added_output = Add()([main_output, branch_path])

    # Flatten and Fully Connected Layers for classification
    flatten = Flatten()(added_output)
    dense1 = Dense(units=128, activation='relu')(flatten)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model