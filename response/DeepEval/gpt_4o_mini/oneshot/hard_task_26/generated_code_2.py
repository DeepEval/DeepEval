import keras
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Add, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 with 3 color channels (RGB)

    # Main Path
    main_path_conv1 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(input_layer)

    # Branch 1: 3x3 Convolution
    branch1 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(main_path_conv1)

    # Branch 2: Max Pooling followed by 3x3 Convolution
    branch2 = MaxPooling2D(pool_size=(2, 2))(main_path_conv1)
    branch2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(branch2)
    branch2 = UpSampling2D(size=(2, 2))(branch2)  # Upsampling back to original size

    # Branch 3: Another Max Pooling followed by 3x3 Convolution
    branch3 = MaxPooling2D(pool_size=(2, 2))(main_path_conv1)
    branch3 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(branch3)
    branch3 = UpSampling2D(size=(2, 2))(branch3)  # Upsampling back to original size

    # Concatenate all branches
    concatenated = Concatenate()([branch1, branch2, branch3])
    main_path_output = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(concatenated)

    # Branch Path
    branch_path_conv1 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(input_layer)

    # Adding the outputs of both paths
    added_output = Add()([main_path_output, branch_path_conv1])

    # Flatten and Fully Connected Layers
    flatten_layer = Flatten()(added_output)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model