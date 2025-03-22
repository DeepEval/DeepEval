import keras
from keras.layers import Input, Conv2D, Concatenate, Add, Flatten, Dense, SeparableConv2D
from keras import regularizers

def dl_model():

    input_layer = Input(shape=(28, 28, 1))
    
    def block(input_tensor):
        conv = SeparableConv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        concat = Concatenate()([input_tensor, conv])
        return concat
    
    main_path = input_layer
    for i in range(3):
        main_path = block(main_path)
        
    branch_path = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same')(input_layer)

    adding_layer = Add()([main_path, branch_path])
    flatten_layer = Flatten()(adding_layer)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model