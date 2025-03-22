import keras
from keras.layers import Input, Conv2D, Add, Concatenate, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 64))
    
    # Main path
    main_path_reduction = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    main_path_conv1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(main_path_reduction)
    main_path_conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(main_path_reduction)
    main_path_concat = Concatenate()([main_path_conv1, main_path_conv2])
    
    # Branch path
    branch_path_conv = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Combine main and branch paths
    combined_path = Add()([main_path_concat, branch_path_conv])
    
    # Fully connected layers
    flatten_layer = Flatten()(combined_path)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model