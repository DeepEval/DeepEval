from keras.models import Model
from keras.layers import Input, BatchNormalization, ReLU, Conv2D, GlobalAveragePooling2D, Dense, Reshape, Multiply, Concatenate, AveragePooling2D


def dl_model():

    input_layer = Input(shape=(32, 32, 3))
    
    growth_rate = 3
    initial_conv = BatchNormalization()(input_layer)
    initial_conv = ReLU()(initial_conv)
    initial_conv = Conv2D(filters=input_layer.shape[-1]*growth_rate, kernel_size=(3, 3), padding='same')(initial_conv)

    input_channels = initial_conv.shape[-1]
    squeeze = GlobalAveragePooling2D()(initial_conv)
    excitation = Dense(units=input_channels // 3, activation='relu')(squeeze) 
    excitation = Dense(units=input_channels, activation='sigmoid')(excitation)
    excitation = Reshape(target_shape=(1, 1, input_channels))(excitation) 
    scaled = Multiply()([initial_conv, excitation])
    se_output = Concatenate()([input_layer, scaled])

    transition = Conv2D(filters=se_output.shape[-1], kernel_size=(1, 1), strides=(1, 1), padding='same')(se_output)
    transition = AveragePooling2D(pool_size=(2, 2), strides=(2, 2))(transition)

    output_layer = Dense(units=10, activation='softmax')(transition)

    model = Model(inputs=input_layer, outputs=output_layer)

    return model
