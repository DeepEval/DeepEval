import keras
from keras.layers import Input, Conv2D, Concatenate, Add, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 64))
    
    # Main path
    main_path_conv1x1 = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    main_path_conv1x1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(main_path_conv1x1)
    main_path_conv3x3 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(main_path_conv1x1)
    
    # Concatenate the outputs of the two parallel convolutions
    main_path_output = Concatenate()([main_path_conv1x1, main_path_conv3x3])

    # Branch path
    branch_path_output = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Ensure the output dimensions match by using Conv2D if needed
    branch_path_output = Conv2D(filters=main_path_output.shape[-1], kernel_size=(1, 1), strides=(1, 1), padding='same')(branch_path_output)

    # Combine the outputs from main and branch paths
    combined_output = Add()([main_path_output, branch_path_output])

    # Flatten and fully connected layers for classification
    flatten_layer = Flatten()(combined_output)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model