import keras
from keras.layers import Input, Conv2D, Concatenate, Add, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 with 3 color channels (RGB)

    # Main pathway with a 1x1 convolution
    main_path_conv = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Parallel branch with 1x1, 1x3, and 3x1 convolutions
    branch_path1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    branch_path2 = Conv2D(filters=32, kernel_size=(1, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    branch_path3 = Conv2D(filters=32, kernel_size=(3, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Concatenate outputs from the parallel branch
    branch_output = Concatenate()([branch_path1, branch_path2, branch_path3])
    
    # Combine main path and branch output
    combined_output = Concatenate()([main_path_conv, branch_output])
    
    # Final 1x1 convolution to match the channel dimension
    final_output = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(combined_output)
    
    # Direct connection for additive fusion
    fusion_output = Add()([final_output, input_layer])

    # Flatten and fully connected layers for classification
    flatten_layer = Flatten()(fusion_output)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)  # 10 classes for CIFAR-10

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model