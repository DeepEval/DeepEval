import keras
from keras.layers import Input, Conv2D, Add, Concatenate, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 with 3 color channels

    # Main pathway
    conv1x1_main = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(input_layer)
    
    # Branch paths
    conv1x1_branch = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(input_layer)
    conv1x3_branch = Conv2D(filters=32, kernel_size=(1, 3), activation='relu', padding='same')(conv1x1_branch)
    conv3x1_branch = Conv2D(filters=32, kernel_size=(3, 1), activation='relu', padding='same')(conv1x1_branch)
    
    # Concatenate branch outputs
    concat_branch = Concatenate()([conv1x1_branch, conv1x3_branch, conv3x1_branch])
    
    # 1x1 convolution on concatenated outputs
    final_conv = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(concat_branch)
    
    # Add direct connection from input to main pathway
    added = Add()([final_conv, input_layer])
    
    # Flatten and pass through fully connected layers
    flatten_layer = Flatten()(added)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)
    return model