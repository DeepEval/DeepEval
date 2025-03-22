import keras
from keras.layers import Input, Conv2D, LayerNormalization, Add, Flatten, Dense, DepthwiseConv2D
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # Main Path
    main_path = DepthwiseConv2D(kernel_size=(7, 7), padding='same', depth_multiplier=1)(input_layer)
    main_path = LayerNormalization()(main_path)
    main_path = Conv2D(filters=input_layer.shape[-1], kernel_size=(1, 1))(main_path)
    main_path = Conv2D(filters=input_layer.shape[-1], kernel_size=(1, 1))(main_path)
    
    # Branch Path
    branch_path = input_layer
    
    # Combine Paths
    combined_path = Add()([main_path, branch_path])
    
    # Flatten and Fully Connected Layers
    flatten_layer = Flatten()(combined_path)
    dense1 = Dense(units=256, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)
    
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model