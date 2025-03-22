import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Add

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 64))
    
    # Main path
    # 1x1 convolutional layer for dimensionality reduction
    main_conv1 = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same')(input_layer)
    # 1x1 convolutional layer for further feature extraction
    main_conv2 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same')(main_conv1)
    # 3x3 convolutional layer for more detailed feature extraction
    main_conv3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same')(main_conv2)
    
    # Flatten and concatenate the outputs from the main path
    main_pool = MaxPooling2D(pool_size=(4, 4))(main_conv3)
    main_flatten = Flatten()(main_pool)
    
    # Branch path
    # 3x3 convolutional layer to match the channel dimensions
    branch_conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same')(input_layer)
    # Concatenate the branch path output with the main path output
    main_branch_concat = Concatenate()([main_flatten, branch_conv1])
    
    # Batch normalization and dense layers
    bn_branch = BatchNormalization()(main_branch_concat)
    dense_branch = Dense(units=128, activation='relu')(bn_branch)
    
    # Add the branch path outputs to the main path outputs
    main_output = Dense(units=10, activation='softmax')(main_flatten)
    branch_output = Dense(units=10, activation='softmax')(dense_branch)
    
    # Model output
    model = keras.Model(inputs=input_layer, outputs=[main_output, branch_output])
    
    return model