from keras.models import Model
from keras.layers import Input, DepthwiseConv2D, GlobalAveragePooling2D, Dense, Reshape, Multiply, Conv2D, Add, Flatten


def dl_model():

    input_layer = Input(shape=(32, 32, 3))
    in_channels = input_layer.shape[-1]

    initial_conv = Conv2D(filters=in_channels*3, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    deepwise_conv2d = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(initial_conv)
    
    channels = deepwise_conv2d.shape[-1]
    gap2d_1= GlobalAveragePooling2D()(deepwise_conv2d)
    dense_1 = Dense(units=channels // 3, activation='relu')(gap2d_1)
    dense_2 = Dense(units=channels, activation='sigmoid')(dense_1)
    reshape = Reshape(target_shape=(1, 1, channels))(dense_2)
    se_output = Multiply()([deepwise_conv2d, reshape])

    conv2 = Conv2D(filters=in_channels, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(se_output)
    final_output = Add()([input_layer, conv2])

    flatten = Flatten()(final_output)
    output_layer = Dense(units=10, activation='softmax')(flatten)

    model = Model(inputs=input_layer, outputs=output_layer)

    return model
