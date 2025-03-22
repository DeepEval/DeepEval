import keras
from keras.layers import Input, Reshape, Permute, Flatten, Dense

def dl_model():     

        input_layer = Input(shape=(32, 32, 3))

        # Reshape the input tensor into three groups
        input_shape = keras.backend.shape(input_layer)
        reshaped_input = Reshape((input_shape[1], input_shape[2], 3, input_shape[3] // 3))(input_layer)

        # Permute the third and fourth dimensions for channel shuffling
        permuted_input = Permute((2, 3, 1, 4))(reshaped_input)

        # Reshape back to original input shape
        reshaped_output = Reshape((input_shape[1], input_shape[2], input_shape[3]))(permuted_input)
        
        # Flatten the output
        flatten_layer = Flatten()(reshaped_output)

        # Dense layer for classification
        output_layer = Dense(units=10, activation='softmax')(flatten_layer)

        model = keras.Model(inputs=input_layer, outputs=output_layer)

        return model