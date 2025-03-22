import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Permute

def dl_model():
    # Define the input shape
    input_shape = (28, 28, 1)
    
    # Block 1
    input_1 = Input(shape=input_shape)
    conv_1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_1)
    dw_conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', data_format='channels_first')(conv_1)
    pointwise_conv_1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(dw_conv)
    pointwise_conv_2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv_1)
    
    concat = Concatenate(axis=-1)([pointwise_conv_1, pointwise_conv_2])
    
    # Block 2
    input_2 = Input(shape=input_shape)
    reshape = Reshape((7, 7, 2))(concat)
    permute = Permute((3, 1, 2))(reshape)
    dw_conv_2 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', data_format='channels_first')(permute)
    reshape_2 = Reshape((7, 7, 2), input_shape=input_shape)(dw_conv_2)
    concat_2 = Concatenate(axis=-1)([reshape_2])
    conv_3 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(concat_2)
    dense = Dense(units=10, activation='softmax')(conv_3)
    
    # Model construction
    model = keras.Model(inputs=[input_1, input_2], outputs=dense)
    
    return model