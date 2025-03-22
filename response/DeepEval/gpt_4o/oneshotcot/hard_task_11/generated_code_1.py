from keras.layers import Input, Conv2D, Concatenate, Add, Flatten, Dense
import keras

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # Main Pathway
    # Path 1: 1x1 Convolution
    path1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Path 2: 1x1 -> 1x3 -> 3x1 Convolutions
    path2_1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    path2_2 = Conv2D(filters=32, kernel_size=(1, 3), strides=(1, 1), padding='same', activation='relu')(path2_1)
    path2_3 = Conv2D(filters=32, kernel_size=(3, 1), strides=(1, 1), padding='same', activation='relu')(path2_2)
    
    # Concatenate Paths
    concat = Concatenate()([path1, path2_3])
    
    # Final 1x1 Convolution to maintain same channel dimensions
    main_output = Conv2D(filters=3, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(concat)
    
    # Additive Fusion with Input
    fused_output = Add()([main_output, input_layer])
    
    # Flatten and Fully Connected Layers for Classification
    flatten_layer = Flatten()(fused_output)
    dense1 = Dense(units=64, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model