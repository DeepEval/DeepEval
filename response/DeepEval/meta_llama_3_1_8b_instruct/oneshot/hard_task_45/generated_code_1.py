import keras
from keras.layers import Input, Lambda, DepthwiseConv2D, Concatenate, BatchNormalization, Flatten, Dense
from keras.layers import Conv2D
from keras import backend as K

def dl_model():     

    input_layer = Input(shape=(32, 32, 3))
    
    # Block 1
    def split_input(input_tensor):
        tensor_split = Lambda(lambda x: K.split(x, 3, axis=-1))(input_tensor)
        return tensor_split
    
    split_input_layer = split_input(input_layer)
    group1 = DepthwiseConv2D(kernel_size=(1, 1), padding='same', activation='relu')(split_input_layer[0])
    group2 = DepthwiseConv2D(kernel_size=(3, 3), padding='same', activation='relu')(split_input_layer[1])
    group3 = DepthwiseConv2D(kernel_size=(5, 5), padding='same', activation='relu')(split_input_layer[2])
    
    output_block1 = Concatenate()([group1, group2, group3])
    
    # Block 2
    branch1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(output_block1)
    branch2 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(output_block1)
    branch2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch2)
    branch3 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(output_block1)
    branch3 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch3)
    branch4 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(output_block1)
    branch4 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(branch4)
    
    output_block2 = Concatenate()([branch1, branch2, branch3, branch4])
    
    batch_norm = BatchNormalization()(output_block2)
    flatten_layer = Flatten()(batch_norm)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model