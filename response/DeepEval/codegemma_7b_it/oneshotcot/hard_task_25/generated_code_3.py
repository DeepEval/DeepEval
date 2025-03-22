import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, AveragePooling2D, UpSampling2D

def dl_model():
    
    input_layer = Input(shape=(32, 32, 3))
    conv = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='sigmoid')(input_layer)
    
    # Main path
    main_path_output = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv)

    # Branch path
    branch_path_output = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv)
    branch_path_output = Conv2D(filters=64, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu')(branch_path_output)
    branch_path_output = UpSampling2D(size=(2, 2))(branch_path_output)
    
    # Concatenate outputs
    concat_output = Concatenate()([main_path_output, branch_path_output])

    # Main path output
    main_path_output = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(concat_output)

    # Branch path output
    branch_path_output = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(concat_output)
    branch_path_output = Conv2D(filters=64, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu')(branch_path_output)
    branch_path_output = UpSampling2D(size=(2, 2))(branch_path_output)
    
    # Fuse outputs
    fused_output = keras.layers.add([main_path_output, branch_path_output])

    # Classification layer
    output_layer = Dense(units=10, activation='softmax')(fused_output)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model