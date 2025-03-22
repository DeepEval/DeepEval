import keras
from keras.layers import Input, Lambda, Conv2D, DepthwiseConv2D, BatchNormalization, Flatten, Dense

def dl_model():
    # Define the input shape
    input_shape = (32, 32, 3)
    
    # Define the main path and branch path
    main_path = Lambda(lambda x: x[:, :, :, :16])(input_layer)
    branch_path = Lambda(lambda x: x[:, :, :, 16:])(input_layer)
    
    # Define the main path layers
    main_path = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same')(main_path)
    main_path = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same')(main_path)
    main_path = BatchNormalization()(main_path)
    main_path = Flatten()(main_path)
    
    # Define the branch path layers
    branch_path = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same')(branch_path)
    branch_path = DepthwiseConv2D(kernel_size=(5, 5), strides=(1, 1), padding='same')(branch_path)
    branch_path = BatchNormalization()(branch_path)
    branch_path = Flatten()(branch_path)
    
    # Add the outputs from the main and branch paths
    merged_path = keras.layers.add([main_path, branch_path])
    
    # Define the fully connected layers
    output_layer = Dense(units=10, activation='softmax')(merged_path)

    # Create the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model