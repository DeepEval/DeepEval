import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Conv2DTranspose, Concatenate, Add, Dense, Flatten

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # Main path
    main_path = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Branch 1: 3x3 convolution
    branch1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(main_path)
    
    # Branch 2: Average pooling, 3x3 convolution, transpose convolution
    branch2 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(main_path)
    branch2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch2)
    branch2 = Conv2DTranspose(filters=64, kernel_size=(2, 2), strides=(2, 2), padding='same', activation='relu')(branch2)
    
    # Branch 3: Average pooling, 3x3 convolution, transpose convolution
    branch3 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(main_path)
    branch3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch3)
    branch3 = Conv2DTranspose(filters=64, kernel_size=(2, 2), strides=(2, 2), padding='same', activation='relu')(branch3)
    
    # Concatenate all branches
    concat_branches = Concatenate()([branch1, branch2, branch3])
    
    # 1x1 convolution on concatenated output
    main_path_output = Conv2D(filters=256, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(concat_branches)
    
    # Branch path: 1x1 convolution
    branch_path = Conv2D(filters=256, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Fuse main path and branch path through addition
    fused_output = Add()([main_path_output, branch_path])
    
    # Flatten and add a fully connected layer for classification
    flatten_layer = Flatten()(fused_output)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)
    
    # Build and return the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model