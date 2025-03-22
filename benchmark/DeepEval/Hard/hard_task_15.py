from keras.models import Model
from keras.layers import Input, GlobalAveragePooling2D, Dense, Reshape, Multiply, Add, Flatten

def dl_model():

    input_layer = Input(shape=(32, 32, 3))
    
    input_channels = input_layer.shape[-1]
    
    squeeze = GlobalAveragePooling2D()(input_layer)
    excitation1 = Dense(units=input_channels // 3, activation='relu')(squeeze) 
    excitation2 = Dense(units=input_channels, activation='sigmoid')(excitation1)
    reshaped = Reshape(target_shape=(1, 1, input_channels))(excitation2)
    main_path = Multiply()([input_layer, reshaped])  
    
    added = Add()([main_path, input_layer])

    flatten_output = Flatten()(added)
    dense_1_output = Dense(units=128, activation='relu')(flatten_output)
    output_layer = Dense(units=10, activation='softmax')(dense_1_output)

    model = Model(inputs=input_layer, outputs=output_layer)

    return model
