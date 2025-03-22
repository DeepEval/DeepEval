from keras.models import Model
from keras.layers import Input, SeparableConv2D, Concatenate, Conv2D, Add, Flatten, Dense,ReLU

def dl_model():

    input_layer = Input(shape=(28, 28, 1))

    def block(input_tensor):

        activation = ReLU()(input_tensor)
        separable_conv = SeparableConv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same')(activation)
        concat = Concatenate()([input_tensor, separable_conv])
        
        return concat
    
    concat_1 = block(input_tensor=input_layer)
    concat_2 = block(input_tensor=concat_1)
    main_path = block(input_tensor=concat_2)

    branch_path = Conv2D(filters=main_path.shape[-1], kernel_size=(3, 3), strides=(1, 1), padding='same')(input_layer)
    added = Add()([branch_path, main_path])
    
    flatten = Flatten()(added)
    output_layer = Dense(units=10, activation='softmax')(flatten)
    
    model = Model(inputs=input_layer, outputs=output_layer)

    return model
