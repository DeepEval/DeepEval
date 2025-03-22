from keras.models import Model
from keras.layers import Input, Conv2D, AveragePooling2D, Add, GlobalAveragePooling2D, Reshape, Multiply, Dense, Flatten

def dl_model():

    input_layer = Input(shape=(32, 32, 3))
    
    def block_1(input_tensor):
        conv1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv2 = Conv2D(filters=input_tensor.shape[-1], kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
        main_path = AveragePooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(conv2)

        output_tensor= Add()([main_path, input_tensor])

        return output_tensor
    
    def block_2(input_tensor):
        squeeze = GlobalAveragePooling2D()(input_tensor)
        input_channels = input_tensor.shape[-1]
        excitation = Dense(units=input_channels // 3, activation='relu')(squeeze) 
        excitation = Dense(units=input_channels, activation='sigmoid')(excitation)
        excitation = Reshape(target_shape=(1, 1, input_channels))(excitation)
        output_tensor = Multiply()([input_tensor, excitation])

        return output_tensor
    
    block_1_output = block_1(input_tensor=input_layer)
    block_2_output = block_2(input_tensor=block_1_output)
    flatten_output = Flatten()(block_2_output)
    output_layer = Dense(units=10, activation='softmax')(flatten_output)

    model = Model(inputs=input_layer, outputs=output_layer)

    return model