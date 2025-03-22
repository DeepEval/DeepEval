import keras
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Add, Concatenate, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # Main path
    initial_conv = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Branch 1
    branch1 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(initial_conv)
    
    # Branch 2
    branch2_pool = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(initial_conv)
    branch2_conv = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(branch2_pool)
    branch2_upsample = UpSampling2D(size=(2, 2))(branch2_conv)
    
    # Branch 3
    branch3_pool = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(initial_conv)
    branch3_conv = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(branch3_pool)
    branch3_upsample = UpSampling2D(size=(2, 2))(branch3_conv)
    
    # Concatenate branches
    concatenated_branches = Concatenate()([branch1, branch2_upsample, branch3_upsample])
    
    # Final 1x1 convolution in the main path
    main_path_output = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(concatenated_branches)
    
    # Branch path
    branch_path_conv = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Add outputs of main and branch paths
    merged_output = Add()([main_path_output, branch_path_conv])
    
    # Fully connected layers
    flatten_layer = Flatten()(merged_output)
    dense1 = Dense(units=256, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)
    
    # Construct and return the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model