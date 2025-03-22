import keras
from keras.layers import Input, Conv2D, GlobalAveragePooling2D, GlobalMaxPooling2D, Add, Multiply, Dense, concatenate, Reshape, Lambda
from keras import backend as K

def channel_attention(input_tensor, name=None):
    channel_axis = -1
    share_axis = 1

    channel = input_tensor.shape[share_axis]
    chanel_attention_vector = Dense(channel, activation='sigmoid')(GlobalAveragePooling2D()(input_tensor))
    chanel_attention_vector = Reshape((1, 1, channel))(chanel_attention_vector)

    chanel_attention_vector = Multiply()([input_tensor, chanel_attention_vector])
    return chanel_attention_vector

def spatial_attention(input_tensor, name=None):
    channel_axis = -1
    share_axis = 1

    spatial = GlobalMaxPooling2D()(input_tensor)
    spatial = Dense(1, activation='sigmoid')(spatial)
    spatial = Reshape((1, 1))(spatial)

    spatial_attention_vector = Multiply()([input_tensor, spatial])
    return spatial_attention_vector

def spatial_channel_attention(input_tensor, name=None):
    channel_attention_vector = channel_attention(input_tensor)
    spatial_attention_vector = spatial_attention(input_tensor)

    channel_attention_vector = Multiply()([channel_attention_vector, spatial_attention_vector])
    return channel_attention_vector

def dl_model():
    
    input_layer = Input(shape=(32, 32, 3))
    conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    path1 = GlobalAveragePooling2D()(conv)
    path1 = Dense(128, activation='relu')(path1)
    path1 = Dense(64, activation='relu')(path1)

    path2 = GlobalMaxPooling2D()(conv)
    path2 = Dense(128, activation='relu')(path2)
    path2 = Dense(64, activation='relu')(path2)

    channel_attention_output = Add()([path1, path2])
    channel_attention_output = Dense(64, activation='relu')(channel_attention_output)
    channel_attention_output = Multiply()([conv, channel_attention_output])

    spatial_attention_output = spatial_channel_attention(conv)
    spatial_attention_output = Multiply()([conv, spatial_attention_output])

    spatial_attention_output = concatenate([channel_attention_output, spatial_attention_output], axis=-1)
    spatial_attention_output = AveragePooling2D(pool_size=(8, 8))(spatial_attention_output)
    spatial_attention_output = MaxPooling2D(pool_size=(8, 8))(spatial_attention_output)
    
    flatten_layer = Flatten()(spatial_attention_output)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model