import keras
from keras.layers import Input, DepthwiseConv2D, LayerNormalization, Conv2D, Concatenate, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # Main path
    main_path = DepthwiseConv2D(kernel_size=(7, 7), strides=(1, 1), padding='same', activation='relu')(input_layer)
    main_path = LayerNormalization(axis=-1, center=True, scale=True)(main_path)
    main_path = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(main_path)
    main_path = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(main_path)
    main_path = Flatten()(main_path)
    
    # Branch path
    branch_path = input_layer
    
    # Combine main and branch paths
    output = Concatenate()([main_path, branch_path])
    output = Flatten()(output)
    output = Dense(units=128, activation='relu')(output)
    output = Dense(units=10, activation='softmax')(output)
    
    model = keras.Model(inputs=input_layer, outputs=output)
    
    return model