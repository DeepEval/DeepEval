import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Conv2DTranspose, Add, Concatenate, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 with 3 color channels

    # Main path
    conv1_1 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(input_layer)
    
    # Branch paths
    branch1 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(conv1_1)
    
    branch2 = AveragePooling2D(pool_size=(2, 2), strides=2)(input_layer)
    branch2 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(branch2)
    branch2 = Conv2DTranspose(filters=32, kernel_size=(2, 2), strides=2, padding='same')(branch2)
    
    branch3 = AveragePooling2D(pool_size=(2, 2), strides=2)(input_layer)
    branch3 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(branch3)
    branch3 = Conv2DTranspose(filters=32, kernel_size=(2, 2), strides=2, padding='same')(branch3)
    
    # Concatenate outputs of branch paths
    concat_branches = Concatenate(axis=-1)([branch1, branch2, branch3])
    
    # Final 1x1 convolutional layer for the main path
    main_path_output = Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(concat_branches)
    
    # Branch path
    branch_path_output = Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(input_layer)
    
    # Add main path and branch path outputs
    added_output = Add()([main_path_output, branch_path_output])
    
    # Flatten and fully connected layer for classification
    flattened = Flatten()(added_output)
    output_layer = Dense(units=10, activation='softmax')(flattened)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model