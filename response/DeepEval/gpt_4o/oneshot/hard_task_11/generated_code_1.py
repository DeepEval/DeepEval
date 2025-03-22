import keras
from keras.layers import Input, Conv2D, Concatenate, Add, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # Main Pathway: First Branch
    path1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Main Pathway: Second Branch
    path2_1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    path2_2 = Conv2D(filters=64, kernel_size=(1, 3), strides=(1, 1), padding='same', activation='relu')(path2_1)
    path2_3 = Conv2D(filters=64, kernel_size=(3, 1), strides=(1, 1), padding='same', activation='relu')(path2_2)
    
    # Concatenate outputs of the two branches
    concatenated = Concatenate()([path1, path2_3])
    
    # Another 1x1 Convolution
    main_output = Conv2D(filters=3, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(concatenated)
    
    # Fusion via an additive operation with the input layer
    fused_output = Add()([main_output, input_layer])
    
    # Classification probabilities through two fully connected layers
    flatten_layer = Flatten()(fused_output)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)
    
    # Define the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model