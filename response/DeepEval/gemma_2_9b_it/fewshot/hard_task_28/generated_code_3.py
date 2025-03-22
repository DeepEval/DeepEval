import keras
from keras.layers import Input, DepthwiseConv2D, Conv2D, LayerNormalization, Add, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # Main Path
    main_path = DepthwiseConv2D(kernel_size=(7, 7), strides=(2, 2), padding='same', activation='relu')(input_layer)
    main_path = LayerNormalization()(main_path)
    main_path = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(main_path)
    main_path = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(main_path)
    
    # Branch Path
    branch_path = input_layer
    
    # Add Branches
    combined_path = Add()([main_path, branch_path])
    
    # Flatten and Fully Connected Layers
    flatten_layer = Flatten()(combined_path)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model