import keras
from keras.layers import Input, Conv2D, Add, Flatten, Dense, Concatenate

def dl_model():
    
    input_layer = Input(shape=(32, 32, 3))

    # Main Pathway
    conv1_1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    branch_path = input_layer
    branch_conv1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(branch_path)
    branch_conv2 = Conv2D(filters=64, kernel_size=(1, 3), strides=(1, 1), padding='same', activation='relu')(branch_path)
    branch_conv3 = Conv2D(filters=64, kernel_size=(3, 1), strides=(1, 1), padding='same', activation='relu')(branch_path)
    
    # Concatenate branches
    branch_output = Concatenate()([branch_conv1, branch_conv2, branch_conv3])
    
    # Add direct connection
    main_path = Add()([conv1_1, branch_output])

    # Final 1x1 convolution
    main_path = Conv2D(filters=3, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(main_path)

    # Flatten and fully connected layers
    flatten_layer = Flatten()(main_path)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)
    return model