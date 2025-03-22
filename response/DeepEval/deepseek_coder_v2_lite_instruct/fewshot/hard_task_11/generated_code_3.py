import keras
from keras.layers import Input, Conv2D, Add, Concatenate, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 with 3 color channels

    # Main pathway
    conv1x1_main = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(input_layer)
    
    # Branch pathway
    conv1x1_branch = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(input_layer)
    conv1x3_branch = Conv2D(filters=32, kernel_size=(1, 3), padding='same', activation='relu')(conv1x1_branch)
    conv3x1_branch = Conv2D(filters=32, kernel_size=(3, 1), padding='same', activation='relu')(conv1x1_branch)
    
    # Concatenate branch outputs
    concatenated = Concatenate()([conv1x1_main, conv1x3_branch, conv3x1_branch])
    
    # Additional 1x1 convolution
    main_output = Conv2D(filters=3, kernel_size=(1, 1), activation='relu')(concatenated)
    
    # Additive operation to fuse with the direct connection
    added = Add()([input_layer, main_output])
    
    # Flatten the result
    flattened = Flatten()(added)
    
    # Fully connected layers
    dense1 = Dense(units=128, activation='relu')(flattened)
    output_layer = Dense(units=10, activation='softmax')(dense1)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model