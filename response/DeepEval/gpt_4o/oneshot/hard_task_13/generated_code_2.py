import keras
from keras.layers import Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Concatenate, Dense, Multiply, Reshape

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # First Block with four parallel paths
    path1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    path2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    path3 = Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(input_layer)
    path4 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(input_layer)

    # Concatenate outputs of the first block
    block1_output = Concatenate()([path1, path2, path3, path4])

    # Second Block
    gap = GlobalAveragePooling2D()(block1_output)
    
    # Two fully connected layers for generating channel-wise weights
    dense1 = Dense(units=block1_output.shape[-1] // 2, activation='relu')(gap)
    dense2 = Dense(units=block1_output.shape[-1], activation='sigmoid')(dense1)

    # Reshape dense2 to match block1_output's shape and multiply element-wise
    weights = Reshape((1, 1, block1_output.shape[-1]))(dense2)
    weighted_feature_map = Multiply()([block1_output, weights])

    # Final fully connected layer for output
    output_layer = Dense(units=10, activation='softmax')(GlobalAveragePooling2D()(weighted_feature_map))

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model