import keras
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate, Add, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # Initial 1x1 convolution in the main path
    main_path = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Branch 1: 3x3 Convolution
    branch1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(main_path)
    
    # Branch 2: Max Pooling -> 3x3 Convolution -> Upsampling
    branch2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(main_path)
    branch2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch2)
    branch2 = UpSampling2D(size=(2, 2))(branch2)
    
    # Branch 3: Max Pooling -> 3x3 Convolution -> Upsampling
    branch3 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(main_path)
    branch3 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch3)
    branch3 = UpSampling2D(size=(2, 2))(branch3)
    
    # Concatenate branches
    concatenated_branches = Concatenate()([branch1, branch2, branch3])
    
    # Final 1x1 Convolution in the main path
    main_output = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(concatenated_branches)
    
    # Branch Path: Initial 1x1 convolution to match channels
    branch_path = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Add outputs of main and branch paths
    combined_output = Add()([main_output, branch_path])
    
    # Fully connected layers for classification
    flatten_layer = Flatten()(combined_output)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model