import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Concatenate, BatchNormalization, Flatten, Dense, Add, Activation, Conv2DTranspose

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Main path
    main_path_conv = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Branch paths
    branch1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(main_path_conv)
    
    branch2 = AveragePooling2D(pool_size=(2, 2), strides=2)(main_path_conv)
    branch2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch2)
    branch2 = Conv2DTranspose(filters=32, kernel_size=(2, 2), strides=(2, 2), padding='same')(branch2)
    
    branch3 = AveragePooling2D(pool_size=(2, 2), strides=2)(main_path_conv)
    branch3 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch3)
    branch3 = Conv2DTranspose(filters=32, kernel_size=(2, 2), strides=(2, 2), padding='same')(branch3)
    
    # Concatenate outputs of branch paths
    concatenated_branches = Concatenate()([branch1, branch2, branch3])
    
    # 1x1 convolutional layer for the main path
    main_path_output = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(concatenated_branches)
    
    # Branch path for channel matching
    branch_path_output = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Fuse main path and branch path outputs through addition
    fused_output = Add()([main_path_output, branch_path_output])
    
    # Flatten the output
    flatten_layer = Flatten()(fused_output)
    
    # Fully connected layers
    dense1 = Dense(units=256, activation='relu')(flatten_layer)
    dense2 = Dense(units=128, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    # Construct and return the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model