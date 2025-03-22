import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():     

    # Define the input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Main path: Conv2D layer 1
    main_path = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Main path: Conv2D layer 2
    main_path = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(main_path)
    
    # Main path: Max pooling layer
    main_path = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(main_path)
    
    # Branch path: Conv2D layer
    branch_path = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Combine the main and branch paths
    combined_path = Concatenate()([main_path, branch_path])
    
    # Add batch normalization layer
    bath_norm = BatchNormalization()(combined_path)
    
    # Flatten the combined features
    flatten_layer = Flatten()(bath_norm)
    
    # Add the first dense layer
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    
    # Add the second dense layer
    output_layer = Dense(units=10, activation='softmax')(dense1)

    # Define the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model