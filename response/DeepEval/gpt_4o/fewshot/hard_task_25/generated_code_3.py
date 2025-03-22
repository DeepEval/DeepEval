import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Conv2DTranspose, Add, Concatenate, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # Main path
    main_1x1_conv = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Branch 1 in the main path
    branch1_conv3x3 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(main_1x1_conv)
    
    # Branch 2 in the main path
    branch2_avgpool = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(main_1x1_conv)
    branch2_conv3x3 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(branch2_avgpool)
    branch2_upsample = Conv2DTranspose(filters=64, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu')(branch2_conv3x3)
    
    # Branch 3 in the main path
    branch3_avgpool = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(main_1x1_conv)
    branch3_conv3x3 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(branch3_avgpool)
    branch3_upsample = Conv2DTranspose(filters=64, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu')(branch3_conv3x3)
    
    # Concatenate the branches
    concat_branches = Concatenate()([branch1_conv3x3, branch2_upsample, branch3_upsample])
    
    # Final 1x1 convolution in the main path
    main_path_output = Conv2D(filters=128, kernel_size=(1, 1), padding='same', activation='relu')(concat_branches)
    
    # Branch path
    branch_1x1_conv = Conv2D(filters=128, kernel_size=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Fuse main and branch paths
    fused_output = Add()([main_path_output, branch_1x1_conv])
    
    # Flatten and fully connected layer for classification
    flatten_layer = Flatten()(fused_output)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model