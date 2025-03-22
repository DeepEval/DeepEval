import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Conv2DTranspose, Concatenate, BatchNormalization, Add, Flatten, Dense

def dl_model():
    
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 with 3 color channels

    # Main path
    main_path_conv1 = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(input_layer)

    # Split into three branches
    branch1 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(main_path_conv1)
    
    branch2 = AveragePooling2D(pool_size=(2, 2))(main_path_conv1)
    branch2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(branch2)
    branch2 = Conv2DTranspose(filters=64, kernel_size=(2, 2), strides=(2, 2), padding='same')(branch2)

    branch3 = AveragePooling2D(pool_size=(4, 4))(main_path_conv1)
    branch3 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(branch3)
    branch3 = Conv2DTranspose(filters=64, kernel_size=(4, 4), strides=(4, 4), padding='same')(branch3)

    # Concatenate outputs of all branches
    concatenated = Concatenate()([branch1, branch2, branch3])
    
    # Final main path output after concatenation
    main_path_output = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(concatenated)

    # Branch path
    branch_path_output = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(input_layer)

    # Fuse main and branch path outputs
    fused_output = Add()([main_path_output, branch_path_output])
    
    # Flatten and fully connected layer
    flatten_layer = Flatten()(fused_output)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)  # 10 classes for CIFAR-10

    # Create model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model