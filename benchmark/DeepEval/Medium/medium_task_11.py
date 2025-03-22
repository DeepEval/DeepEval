from keras.models import Model
from keras.layers import Input, GlobalAveragePooling2D, Dense, GlobalMaxPooling2D, Activation,AveragePooling2D,MaxPooling2D,Reshape, Multiply, Concatenate, Conv2D, Add, Flatten

def dl_model():

    input_layer = Input(shape=(32, 32, 3))
    
    input_channels = input_layer.shape[-1]
    initial_conv = Conv2D(filters=input_channels, kernel_size=(1, 1), strides=(1, 1), padding='same')(input_layer)
    
    global_avgpool = GlobalAveragePooling2D()(initial_conv)
    avg_out = Dense(units=input_channels // 3, activation='relu')(global_avgpool)
    avg_out = Dense(units=input_channels, activation='relu')(avg_out)

    global_maxpool = GlobalMaxPooling2D()(initial_conv)
    max_out = Dense(units=input_channels // 3, activation='relu')(global_maxpool)
    max_out = Dense(units=input_channels, activation='relu')(max_out)
    channel = Add()([avg_out, max_out])
    channel = Activation(activation='sigmoid')(channel)
    channel = Reshape(target_shape=(1, 1, input_channels))(channel)
    channel_out = Multiply()([initial_conv, channel])

    avgpool = AveragePooling2D(pool_size=(2, 2), strides=(1, 1), padding='same')(channel_out)
    maxpool = MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same')(channel_out)
    spatial = Concatenate()([avgpool, maxpool])
    spatial_out = Conv2D(filters=3, kernel_size=(7, 7), strides=(1, 1), padding='same', activation='sigmoid')(spatial)
    
    scaled = Multiply()([channel_out, spatial_out])

    flatten_output = Flatten()(scaled)
    dense = Dense(units=128, activation='relu')(flatten_output)
    output_layer = Dense(units=10, activation='softmax')(dense)

    model = Model(inputs=input_layer, outputs=output_layer)

    return model
