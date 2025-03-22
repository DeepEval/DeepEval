import keras
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate, Add, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # Main path
    main_conv1x1 = Conv2D(filters=64, kernel_size=(1, 1), activation='relu', padding='same')(input_layer)
    
    # Branch 1
    branch1_conv3x3 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(main_conv1x1)
    
    # Branch 2
    branch2_maxpool = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(main_conv1x1)
    branch2_conv3x3 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(branch2_maxpool)
    branch2_upsample = UpSampling2D(size=(2, 2))(branch2_conv3x3)
    
    # Branch 3
    branch3_maxpool = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(main_conv1x1)
    branch3_conv3x3 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(branch3_maxpool)
    branch3_upsample = UpSampling2D(size=(2, 2))(branch3_conv3x3)
    
    # Concatenate branches
    concatenated_branches = Concatenate()([branch1_conv3x3, branch2_upsample, branch3_upsample])
    
    # Main path final 1x1 convolution
    main_output_conv1x1 = Conv2D(filters=64, kernel_size=(1, 1), activation='relu', padding='same')(concatenated_branches)
    
    # Branch path
    branch_path_conv1x1 = Conv2D(filters=64, kernel_size=(1, 1), activation='relu', padding='same')(input_layer)
    
    # Add main path and branch path
    added_paths = Add()([main_output_conv1x1, branch_path_conv1x1])
    
    # Fully connected layers for classification
    flatten = Flatten()(added_paths)
    dense1 = Dense(units=128, activation='relu')(flatten)
    output_layer = Dense(units=10, activation='softmax')(dense1)
    
    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model