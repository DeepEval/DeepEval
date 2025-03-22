import keras
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate, Add, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Main path
    initial_conv_main = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(input_layer)

    # Branch 1: 3x3 Convolution
    branch1 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(initial_conv_main)

    # Branch 2: Max Pooling, then 3x3 Convolution, Upsample
    branch2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(initial_conv_main)
    branch2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(branch2)
    branch2 = UpSampling2D(size=(2, 2))(branch2)

    # Branch 3: Max Pooling, then 3x3 Convolution, Upsample
    branch3 = MaxPooling2D(pool_size=(4, 4), strides=(4, 4), padding='same')(initial_conv_main)
    branch3 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(branch3)
    branch3 = UpSampling2D(size=(4, 4))(branch3)

    # Concatenate all branches
    concatenated = Concatenate()([branch1, branch2, branch3])

    # Final convolution in main path
    main_path_output = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(concatenated)

    # Branch path
    initial_conv_branch = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(input_layer)

    # Adding outputs from main path and branch path
    added_output = Add()([main_path_output, initial_conv_branch])

    # Fully connected layers for classification
    flatten_layer = Flatten()(added_output)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model