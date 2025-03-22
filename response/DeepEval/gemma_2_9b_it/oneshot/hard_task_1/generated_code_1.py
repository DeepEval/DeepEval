import keras
from keras.layers import Input, Conv2D, GlobalAveragePooling2D, GlobalMaxPooling2D, Flatten, Dense, Concatenate, Add, Activation, Multiply

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Block 1: Channel Attention
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)

    avg_pool = GlobalAveragePooling2D()(conv1)
    dense1 = Dense(units=128, activation='relu')(avg_pool)
    dense2 = Dense(units=3, activation='sigmoid')(dense1)  # Output channels match input channels

    max_pool = GlobalMaxPooling2D()(conv1)
    dense3 = Dense(units=128, activation='relu')(max_pool)
    dense4 = Dense(units=3, activation='sigmoid')(dense3)

    channel_attention = Add()([dense2, dense4]) 

    # Apply channel attention weights
    channel_output = Multiply()([conv1, channel_attention])

    # Block 2: Spatial Feature Extraction
    avg_pool2 = keras.layers.AveragePooling2D(pool_size=(2, 2))(channel_output)
    max_pool2 = keras.layers.MaxPooling2D(pool_size=(2, 2))(channel_output)

    concat_spatial = Concatenate(axis=3)([avg_pool2, max_pool2])
    conv2 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='sigmoid')(concat_spatial)
    normalized_spatial = Activation('sigmoid')(conv2)

    # Combine channel and spatial features
    combined_features = Multiply()([normalized_spatial, channel_output])

    # Final Branch and Classification
    branch = Conv2D(filters=3, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(combined_features)
    output = Add()([branch, combined_features])
    output = Flatten()(output)
    output_layer = Dense(units=10, activation='softmax')(output)

    model = keras.Model(inputs=input_layer, outputs=output_layer)
    return model