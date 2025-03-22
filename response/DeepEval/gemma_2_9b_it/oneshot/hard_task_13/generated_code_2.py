import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, GlobalAveragePooling2D, Flatten, Dense, Reshape

def dl_model():     

    input_layer = Input(shape=(32, 32, 3))  

    def block1(input_tensor):

        conv1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv3 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        maxpool = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(input_tensor)

        output_tensor = Concatenate()([conv1, conv2, conv3, maxpool])

        return output_tensor

    block1_output = block1(input_layer)

    block2_input = GlobalAveragePooling2D()(block1_output)

    dense1 = Dense(units=block2_input.shape[-1], activation='relu')(block2_input)
    dense2 = Dense(units=block2_input.shape[-1], activation='relu')(dense1)
    reshaped_weights = Reshape(block1_output.shape[1:3])(dense2)

    weighted_output = keras.layers.Multiply()([block1_output, reshaped_weights]) 
    output_layer = Dense(units=10, activation='softmax')(weighted_output)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model