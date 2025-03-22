import keras
from keras.layers import Input, Conv2D, DepthwiseConv2D, Concatenate, BatchNormalization, GlobalAveragePooling2D, Reshape, Multiply, Dense, Flatten
from keras.regularizers import l2

def dl_model():     

    input_layer = Input(shape=(32, 32, 3))
    conv1x1 = Conv2D(filters=3*3, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    depthwise_conv = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1x1)
    
    channel_attention = GlobalAveragePooling2D()(depthwise_conv)
    channel_attention = Dense(units=3*3, activation='relu', kernel_regularizer=l2(0.01))(channel_attention)
    channel_attention = Dense(units=3*3, activation='sigmoid', kernel_regularizer=l2(0.01))(channel_attention)
    channel_attention = Reshape(target_shape=(1, 1, 3*3))(channel_attention)
    
    channel_attention_weighted = Multiply()([depthwise_conv, channel_attention])
    
    conv1x1_reduce = Conv2D(filters=3, kernel_size=(1, 1), strides=(1, 1), padding='same')(channel_attention_weighted)
    combined = Concatenate()([conv1x1_reduce, input_layer])
    
    batch_norm = BatchNormalization()(combined)
    flatten_layer = Flatten()(batch_norm)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model