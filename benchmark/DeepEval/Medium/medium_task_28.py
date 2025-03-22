from keras.models import Model
from keras.layers import Input, Conv2D, Softmax, Multiply, LayerNormalization, ReLU, Add, Flatten, Dense

def dl_model():

    input_layer = Input(shape=(32, 32, 3))
    
    def block(input_tensor):
        
        input_channels = input_tensor.shape[-1]
        conv_mask = Conv2D(filters=input_channels, kernel_size=(1, 1), strides=(1, 1), padding='same')(input_tensor) 
        softmax = Softmax()(conv_mask)
        context = Multiply()([input_tensor, softmax])
        
        conv = Conv2D(filters=input_channels // 3, kernel_size=(1, 1), strides=(1, 1), padding='same')(context) 
        norm = LayerNormalization()(conv)
        act = ReLU()(norm)
        
        main_path = Conv2D(filters=input_channels, kernel_size=(1, 1), strides=(1, 1), padding='same')(act)
        output_tensor = Add()([input_tensor, main_path])

        return output_tensor

    block_output = block(input_tensor=input_layer)
    flatten_output = Flatten()(block_output)
    dense_1 = Dense(units=128, activation='relu')(flatten_output)
    output_layer = Dense(units=10, activation='softmax')(dense_1)

    model = Model(inputs=input_layer, outputs=output_layer)

    return model
