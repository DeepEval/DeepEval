import keras
from keras.layers import Input, Conv2D, GlobalAveragePooling2D, Flatten, Dense, Reshape, multiply

def dl_model():     

    input_layer = Input(shape=(32, 32, 3))

    # Increase channels threefold
    x = Conv2D(filters=3 * input_layer.shape[-1], kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Initial feature extraction with depthwise separable convolution
    x = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', depth_multiplier=1)(x)

    # Channel attention block
    channel_attention_weights = GlobalAveragePooling2D()(x)
    channel_attention_weights = Dense(units=32, activation='relu')(channel_attention_weights)
    channel_attention_weights = Dense(units=x.shape[-1], activation='sigmoid')(channel_attention_weights)
    channel_attention_weights = Reshape(target_shape=(1, 1, channel_attention_weights.shape[-1]))(channel_attention_weights)
    x = multiply([x, channel_attention_weights])  

    # Reduce dimensionality
    x = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(x)

    # Element-wise addition with initial input
    x = keras.layers.add([input_layer, x])

    x = Flatten()(x)
    output_layer = Dense(units=10, activation='softmax')(x)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model