import keras
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate, Add, Flatten, Dense

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Main path
    main_initial_conv = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Branch 1: 3x3 convolution
    branch1 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(main_initial_conv)
    
    # Branch 2: Max pooling, 3x3 convolution, and upsampling
    branch2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(main_initial_conv)
    branch2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(branch2)
    branch2 = UpSampling2D(size=(2, 2))(branch2)
    
    # Branch 3: Max pooling, 3x3 convolution, and upsampling
    branch3 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(main_initial_conv)
    branch3 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(branch3)
    branch3 = UpSampling2D(size=(2, 2))(branch3)
    
    # Concatenate branches
    main_concat = Concatenate()([branch1, branch2, branch3])
    
    # 1x1 convolution after concatenation
    main_output = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(main_concat)
    
    # Branch path
    branch_path = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Add main path and branch path
    combined = Add()([main_output, branch_path])
    
    # Flatten and dense layers for classification
    flatten_layer = Flatten()(combined)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)
    
    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model