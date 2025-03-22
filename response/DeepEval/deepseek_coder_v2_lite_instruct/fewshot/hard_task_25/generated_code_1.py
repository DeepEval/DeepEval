import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Conv2DTranspose, Add, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 with 3 color channels

    # Main path
    main_path_input = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Branch paths
    branch1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(main_path_input)
    
    branch2 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(input_layer)
    branch2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch2)
    branch2 = Conv2DTranspose(filters=32, kernel_size=(2, 2), strides=(2, 2), padding='same')(branch2)
    
    branch3 = AveragePooling2D(pool_size=(4, 4), strides=(4, 4), padding='same')(input_layer)
    branch3 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch3)
    branch3 = Conv2DTranspose(filters=32, kernel_size=(4, 4), strides=(4, 4), padding='same')(branch3)
    
    # Concatenate outputs of branch paths
    concatenated_branches = Add()([branch1, branch2, branch3])
    
    # 1x1 convolutional layer to form the main path output
    main_path_output = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(concatenated_branches)
    
    # Branch path to match the number of channels of the main path
    branch_path_input = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Fuse main path and branch path outputs through addition
    fused_output = Add()([main_path_output, branch_path_input])
    
    # Flatten and fully connected layer for classification
    flattened_output = Flatten()(fused_output)
    final_output = Dense(units=10, activation='softmax')(flattened_output)
    
    model = keras.Model(inputs=input_layer, outputs=final_output)
    
    return model