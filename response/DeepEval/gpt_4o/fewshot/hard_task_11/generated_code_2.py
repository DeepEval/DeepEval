import keras
from keras.layers import Input, Conv2D, Add, Concatenate, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images have 32x32 size with 3 color channels

    # Main pathway
    main_path_conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Parallel branch with 1x1, 1x3, and 3x1 convolutions
    branch_conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    branch_conv2 = Conv2D(filters=32, kernel_size=(1, 3), strides=(1, 1), padding='same', activation='relu')(branch_conv1)
    branch_conv3 = Conv2D(filters=32, kernel_size=(3, 1), strides=(1, 1), padding='same', activation='relu')(branch_conv2)
    
    # Concatenate outputs of main and branch pathways
    concatenated = Concatenate()([main_path_conv1, branch_conv3])
    
    # Further process concatenated output
    main_output = Conv2D(filters=3, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(concatenated)
    
    # Direct connection from input to the main output through an additive operation
    fusion = Add()([input_layer, main_output])
    
    # Flatten and fully connected layers for classification
    flatten_layer = Flatten()(fusion)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model