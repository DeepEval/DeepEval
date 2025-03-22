import keras
from keras.layers import Input, Conv2D, AveragePooling2D, UpSampling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    
    input_layer = Input(shape=(32, 32, 3))
    
    # Main Path
    x = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    x_1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x)
    x_2 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
    x_2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x_2)
    x_2 = UpSampling2D(size=(2, 2))(x_2)
    x_3 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
    x_3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x_3)
    x_3 = UpSampling2D(size=(2, 2))(x_3)
    
    main_path_output = Concatenate()([x_1, x_2, x_3])
    main_path_output = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(main_path_output)

    # Branch Path
    branch_path = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Fusion
    output = keras.layers.add([main_path_output, branch_path])
    
    output = Flatten()(output)
    output = Dense(units=10, activation='softmax')(output)
    
    model = keras.Model(inputs=input_layer, outputs=output)
    return model