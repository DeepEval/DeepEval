import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():     

    input_layer = Input(shape=(28, 28, 1))
    
    def separable_conv_block(input_tensor):
        x = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        x = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same')(x)
        return x

    main_path = input_layer
    for _ in range(3):
        main_path = separable_conv_block(main_path)
    
    branch_path = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    merged_features = keras.layers.Add()([main_path, branch_path])  
    
    bath_norm = BatchNormalization()(merged_features)
    flatten_layer = Flatten()(bath_norm)
    dense_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=dense_layer)

    return model