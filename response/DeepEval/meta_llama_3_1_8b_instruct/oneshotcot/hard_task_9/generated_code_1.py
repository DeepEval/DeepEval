import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Add

def dl_model():     

    input_layer = Input(shape=(32, 32, 3))
    
    # Feature extraction branch 1: 1x1 convolution
    branch1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Feature extraction branch 2: 1x1 convolution + 3x3 convolution
    branch2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    branch2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch2)

    # Feature extraction branch 3: 1x1 convolution + 3x3 convolution + 3x3 convolution
    branch3 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    branch3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch3)
    branch3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch3)

    # Concatenate the outputs from the three branches
    output_branch = Concatenate()([branch1, branch2, branch3])

    # Adjust the output dimensions to match the input image's channel size
    output_branch = Conv2D(filters=3, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(output_branch)

    # Direct connection to the input
    direct_connect = input_layer

    # Fuse the main path and the branch together through addition
    fused_output = Add()([output_branch, direct_connect])

    # Apply batch normalization and flatten the result
    bath_norm = BatchNormalization()(fused_output)
    flatten_layer = Flatten()(bath_norm)

    # Add three fully connected layers to produce the final classification
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model