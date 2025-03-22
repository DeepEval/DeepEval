import keras
from keras.layers import Input, Conv2D, AveragePooling2D, UpSampling2D, Add, Concatenate, BatchNormalization, Flatten, Dense
from keras.models import Model

def dl_model():
    
    input_layer = Input(shape=(32, 32, 3))
    
    # Main path
    main_path = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Branch paths
    # First branch: 3x3 convolution
    branch1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(main_path)
    
    # Second branch: average pooling + 3x3 convolution + upsampling
    branch2 = AveragePooling2D(pool_size=(2, 2))(main_path)
    branch2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch2)
    branch2 = UpSampling2D(size=(2, 2))(branch2)

    # Third branch: average pooling + 3x3 convolution + upsampling
    branch3 = AveragePooling2D(pool_size=(2, 2))(main_path)
    branch3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch3)
    branch3 = UpSampling2D(size=(2, 2))(branch3)

    # Concatenate outputs from all branches
    concatenated_output = Concatenate()([branch1, branch2, branch3])
    
    # Apply 1x1 convolution to the concatenated output
    main_path_output = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(concatenated_output)
    
    # Branch path
    branch_path_output = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Fuse main path and branch path outputs
    final_output = Add()([main_path_output, branch_path_output])
    
    # Batch normalization and flatten
    bath_norm = BatchNormalization()(final_output)
    flatten_layer = Flatten()(bath_norm)
    
    # Fully connected layer for classification
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = Model(inputs=input_layer, outputs=output_layer)

    return model