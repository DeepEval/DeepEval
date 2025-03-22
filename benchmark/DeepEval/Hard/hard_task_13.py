from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, GlobalAveragePooling2D, Reshape, Multiply, Dense

def dl_model():

    input_layer = Input(shape=(32, 32, 3))
    
    def block_1(input_tensor):
        
        conv1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(2, 2), padding='same', activation='relu')(input_tensor)
        conv3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu')(input_tensor)
        conv5 = Conv2D(filters=64, kernel_size=(5, 5), strides=(2, 2), padding='same', activation='relu')(input_tensor)
        pool = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(input_tensor)
        output_tensor = Concatenate()([conv1, conv3, conv5, pool])

        return output_tensor
    
    def block_2(input_tensor):
        
        channels = input_tensor.shape[-1]
        squeeze = GlobalAveragePooling2D()(input_tensor)
        excitation1 = Dense(units=channels // 16, activation='relu')(squeeze)
        excitation2 = Dense(units=channels, activation='sigmoid')(excitation1)
        reshaped = Reshape(target_shape=(1, 1, channels))(excitation2)
        output_tensor = Multiply()([input_tensor, reshaped])

        return output_tensor
    
    block_1_output = block_1(input_tensor=input_layer)
    block_2_output = block_2(input_tensor=block_1_output)
    output_layer = Dense(units=10, activation='softmax')(block_2_output)

    model = Model(inputs=input_layer, outputs=output_layer)

    return model
