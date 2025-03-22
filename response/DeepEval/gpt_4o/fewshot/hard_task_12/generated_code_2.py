import keras
from keras.layers import Input, Conv2D, Concatenate, Add, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 64))
    
    # Main path
    main_path_1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Two parallel convolutional layers in the main path
    main_path_2a = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(main_path_1)
    main_path_2b = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(main_path_1)
    main_path_concat = Concatenate()([main_path_2a, main_path_2b])
    
    # Branch path
    branch_path = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Combine the outputs of the main path and branch path
    combined = Add()([main_path_concat, branch_path])
    
    # Flatten and fully connected layers for classification
    flatten_layer = Flatten()(combined)
    fc1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(fc1)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model