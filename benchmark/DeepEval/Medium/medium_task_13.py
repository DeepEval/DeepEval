from keras.models import Model
from keras.layers import Input, Conv2D, Concatenate, Flatten, Dense

def dl_model():

    input_layer = Input(shape=(32, 32, 3))
    
    def block(input_tensor):
        conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1,1),padding='same',activation='relu')(input_tensor)
        conv1 = Concatenate()([input_tensor, conv1])

        conv2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1,1),padding='same',activation='relu')(conv1)
        conv2 = Concatenate()([conv2, conv1])

        conv3 = Conv2D(filters=32, kernel_size=(3, 3),strides=(1,1),padding='same',activation='relu')(conv2)
        output_tensor = Concatenate()([conv3, conv2])

        return output_tensor
    
    block_output = block(input_tensor=input_layer)
    flatten_output = Flatten()(block_output)
    dense_1_output = Dense(units=128, activation='relu')(flatten_output)
    output_layer = Dense(units=10, activation='softmax')(dense_1_output)

    model = Model(inputs=input_layer, outputs=output_layer)

    return model
