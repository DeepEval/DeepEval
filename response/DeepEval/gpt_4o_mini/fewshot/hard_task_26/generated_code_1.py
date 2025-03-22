import keras
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate, Add, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 pixels with 3 color channels (RGB)

    # Main Path
    conv1 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(input_layer)

    # Branch 1: 3x3 Convolution
    branch1 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(conv1)

    # Branch 2: Max Pooling followed by 3x3 Convolution
    branch2_pool = MaxPooling2D(pool_size=(2, 2))(conv1)
    branch2_conv = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(branch2_pool)
    branch2_up = UpSampling2D(size=(2, 2))(branch2_conv)

    # Branch 3: Another Max Pooling followed by 3x3 Convolution
    branch3_pool = MaxPooling2D(pool_size=(4, 4))(conv1)
    branch3_conv = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(branch3_pool)
    branch3_up = UpSampling2D(size=(4, 4))(branch3_conv)

    # Concatenate outputs from all branches
    concatenated = Concatenate()([branch1, branch2_up, branch3_up])

    # Final 1x1 Convolution in the main path
    main_path_output = Conv2D(filters=128, kernel_size=(1, 1), padding='same', activation='relu')(concatenated)

    # Branch Path
    branch_path_conv = Conv2D(filters=128, kernel_size=(1, 1), padding='same', activation='relu')(input_layer)

    # Add outputs from the main path and branch path
    added_output = Add()([main_path_output, branch_path_conv])

    # Flatten and Fully Connected layers for classification
    flatten_layer = Flatten()(added_output)
    dense1 = Dense(units=256, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model