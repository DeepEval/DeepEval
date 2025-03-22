import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Conv2DTranspose, Add, Concatenate, Dense, Flatten

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # Main path
    main_path = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # First branch: 3x3 Convolution
    branch1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(main_path)
    
    # Second branch: Average Pooling -> 3x3 Convolution -> Transpose Convolution
    branch2 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(main_path)
    branch2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch2)
    branch2 = Conv2DTranspose(filters=64, kernel_size=(2, 2), strides=(2, 2), padding='same', activation='relu')(branch2)
    
    # Third branch: Average Pooling -> 3x3 Convolution -> Transpose Convolution
    branch3 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(main_path)
    branch3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch3)
    branch3 = Conv2DTranspose(filters=64, kernel_size=(2, 2), strides=(2, 2), padding='same', activation='relu')(branch3)
    
    # Concatenate outputs from branches
    concatenated_branches = Concatenate()([branch1, branch2, branch3])
    
    # 1x1 Convolution after concatenation
    main_output = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(concatenated_branches)
    
    # Branch path: 1x1 Convolution
    branch_path = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Fuse main path and branch path
    fused_output = Add()([main_output, branch_path])
    
    # Flatten and Fully Connected Layer for classification
    flatten = Flatten()(fused_output)
    output_layer = Dense(units=10, activation='softmax')(flatten)
    
    # Build the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model