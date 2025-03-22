import keras
from keras.layers import Input, Conv2D, GlobalAveragePooling2D, Flatten, Dense, Reshape

def dl_model():

    input_layer = Input(shape=(32, 32, 3))
    
    # Increase channels threefold
    conv1 = Conv2D(filters=3 * input_layer.shape[2], kernel_size=(1, 1), padding='same', activation='relu')(input_layer)

    # Extract initial features
    depthwise_conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', depth_multiplier=1)(conv1) 

    # Channel attention
    channel_attn = GlobalAveragePooling2D()(depthwise_conv)
    channel_attn = Dense(units=32, activation='relu')(channel_attn)
    channel_attn = Dense(units=32 * depthwise_conv.shape[3], activation='sigmoid')(channel_attn)
    channel_attn = Reshape(target_shape=(32, depthwise_conv.shape[3]))(channel_attn)

    # Apply channel attention weights
    output = conv1 * channel_attn 

    # Reduce dimensionality
    output = Conv2D(filters=16, kernel_size=(1, 1), padding='same', activation='relu')(output)

    # Combine with initial input
    output = keras.layers.add([output, conv1])

    # Flatten and classification
    flatten = Flatten()(output)
    output_layer = Dense(units=10, activation='softmax')(flatten)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model