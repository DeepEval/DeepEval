import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Conv2DTranspose, Add, Flatten, Dense, Concatenate

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Main path
    conv_main_1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Branch paths
    conv_branch_1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    avg_pool_1 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(input_layer)
    conv_branch_2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(avg_pool_1)
    conv_branch_2_trans = Conv2DTranspose(filters=32, kernel_size=(2, 2), strides=(2, 2), padding='same')(conv_branch_2)
    
    avg_pool_2 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(input_layer)
    conv_branch_3 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(avg_pool_2)
    conv_branch_3_trans = Conv2DTranspose(filters=32, kernel_size=(2, 2), strides=(2, 2), padding='same')(conv_branch_3)
    
    # Concatenate outputs of branch paths
    branch_output = Concatenate()([conv_branch_1, conv_branch_2_trans, conv_branch_3_trans])
    
    # 1x1 convolutional layer to match the number of channels in the main path
    conv_branch_final = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(branch_output)
    
    # Add main path and branch path outputs
    added_output = Add()([conv_main_1, conv_branch_final])
    
    # Flatten the output
    flatten_layer = Flatten()(added_output)
    
    # Fully connected layer for classification
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)
    
    # Create the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model