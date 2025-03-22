import keras
from keras.layers import Input, Conv2D, GlobalAveragePooling2D, Flatten, Dense, Reshape, Multiply

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Increase channel dimensionality
    x = Conv2D(filters=input_layer.shape[-1] * 3, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Initial feature extraction
    x = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', depth_wise=True)(x)

    # Channel attention module
    channel_attention = GlobalAveragePooling2D()(x)
    channel_attention = Dense(units=32, activation='relu')(channel_attention)
    channel_attention = Dense(units=x.shape[-1], activation='sigmoid')(channel_attention)
    channel_attention = Reshape(target_shape=(x.shape[-1], 1, 1))(channel_attention)

    # Apply channel attention weighting
    x = Multiply()([x, channel_attention])

    # Reduce dimensionality
    x = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(x)

    # Residual connection
    x = keras.layers.add([x, input_layer])

    # Classification
    x = Flatten()(x)
    output_layer = Dense(units=10, activation='softmax')(x)

    model = keras.Model(inputs=input_layer, outputs=output_layer)
    return model