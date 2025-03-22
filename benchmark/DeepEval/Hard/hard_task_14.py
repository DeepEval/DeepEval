from keras.models import Model
from keras.layers import Input, GlobalAveragePooling2D, Dense, Reshape, Multiply, Conv2D, Add

def dl_model():

    input_layer = Input(shape=(32, 32, 3))
    
    input_channels = input_layer.shape[-1]
    
    squeeze = GlobalAveragePooling2D()(input_layer)
    excitation1 = Dense(units=input_channels // 3, activation='relu')(squeeze) 
    excitation2 = Dense(units=input_channels, activation='sigmoid')(excitation1)
    reshaped = Reshape(target_shape=(1, 1, input_channels))(excitation2)
    main_path = Multiply()([input_layer, reshaped])  

    branch_path = Conv2D(filters=input_channels, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    added = Add()([main_path, branch_path])

    dense_1 = Dense(units=128, activation='relu')(added)
    dense_2 = Dense(units=64, activation='relu')(dense_1)
    output_layer = Dense(units=10, activation='softmax')(dense_2)

    model = Model(inputs=input_layer, outputs=output_layer)

    return model
