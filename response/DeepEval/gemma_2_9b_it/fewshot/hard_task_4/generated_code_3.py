import keras
from keras.layers import Input, Conv2D, GlobalAveragePooling2D, Dense, Reshape, Add

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Increase channel dimensionality
    conv1 = Conv2D(filters=input_layer.shape[-1] * 3, kernel_size=(1, 1), strides=(1, 1), padding='same')(input_layer)

    # Extract initial features
    depthwise_conv = Conv2D(filters=input_layer.shape[-1], kernel_size=(3, 3), strides=(1, 1), padding='same', 
                           activation='relu', depth_multiplier=1)(conv1)

    # Channel Attention
    channel_pool = GlobalAveragePooling2D()(depthwise_conv)
    dense1 = Dense(units=channel_pool.shape[-1]//2, activation='relu')(channel_pool)
    dense2 = Dense(units=input_layer.shape[-1], activation='sigmoid')(dense1)
    attention_weights = Reshape((input_layer.shape[-1],))(dense2)

    # Apply channel attention weighting
    weighted_features = tf.multiply(depthwise_conv, attention_weights)

    # Reduce dimensionality
    conv2 = Conv2D(filters=input_layer.shape[-1], kernel_size=(1, 1), strides=(1, 1), padding='same')(weighted_features)

    # Combine with original input
    output = Add()([conv2, conv1])

    # Flatten and classify
    flatten = Flatten()(output)
    output_layer = Dense(units=10, activation='softmax')(flatten)

    model = keras.Model(inputs=input_layer, outputs=output_layer)
    return model