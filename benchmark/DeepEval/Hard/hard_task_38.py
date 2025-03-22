from keras.models import Model
from keras.layers import Input, BatchNormalization, Conv2D, Concatenate, Dense, ReLU

def dl_model():
    
    input_layer = Input(shape=(28,28,1))

    def block(input_tensor):

        batchnormal = BatchNormalization()(input_tensor)
        activated = ReLU()(batchnormal)
        conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same')(activated)
        output_tensor = Concatenate()([input_tensor, conv])

        return output_tensor

    path1_block1_output = block(input_tensor=input_layer)
    path1_block2_output = block(input_tensor=path1_block1_output)
    path1_block3_output = block(input_tensor=path1_block2_output)
    path1 = path1_block3_output

    path2_block1_output = block(input_tensor=input_layer)
    path2_block2_output = block(input_tensor=path2_block1_output)
    path2_block3_output = block(input_tensor=path2_block2_output)
    path2 = path2_block3_output

    concatenated = Concatenate()([path1, path2])

    dense = Dense(units=64, activation='relu')(concatenated)
    output_layer = Dense(units=10, activation='softmax')(dense)
    
    model = Model(inputs=input_layer, outputs=output_layer)

    return model

