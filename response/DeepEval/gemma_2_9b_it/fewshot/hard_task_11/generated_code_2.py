import keras
from keras.layers import Input, Conv2D, Add, Concatenate, Flatten, Dense

def dl_model():

    input_layer = Input(shape=(32, 32, 3))

    # Main pathway
    conv1x1_main = Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(input_layer)
    
    # Parallel branch
    conv1x1_branch = Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(input_layer)
    conv1x3_branch = Conv2D(filters=64, kernel_size=(1, 3), activation='relu')(input_layer)
    conv3x1_branch = Conv2D(filters=64, kernel_size=(3, 1), activation='relu')(input_layer)

    # Concatenate outputs from parallel branch
    branch_output = Concatenate()([conv1x1_branch, conv1x3_branch, conv3x1_branch])

    # Combine main and branch outputs
    combined_output = Add()([conv1x1_main, branch_output])
    
    # Final 1x1 convolution
    output = Conv2D(filters=3, kernel_size=(1, 1), activation='relu')(combined_output)

    # Flatten and dense layers
    flatten = Flatten()(output)
    dense1 = Dense(units=128, activation='relu')(flatten)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model