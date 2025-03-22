from keras.models import Model
from keras.layers import Input, Conv2D, Add, Concatenate, Flatten, Dense

def dl_model():
    
    input_layer = Input(shape=(28,28,1))

    def block(input_tensor):
        
        main_path_1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        main_path_2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(main_path_1)
        main_path_3 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(main_path_2)

        branch_path = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        output_tensor = Add()([main_path_1, main_path_2, main_path_3, branch_path])

        return output_tensor

    block_output_1 = block(input_tensor=input_layer)
    block_output_2 = block(input_tensor=input_layer)
    concatenated = Concatenate()([block_output_1, block_output_2])
    
    flatten = Flatten()(concatenated)
    output_layer = Dense(units=10, activation='softmax')(flatten)
    
    model = Model(inputs=input_layer, outputs=output_layer)

    return model
