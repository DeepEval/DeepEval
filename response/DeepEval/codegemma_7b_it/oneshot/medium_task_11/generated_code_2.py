import keras
from keras.layers import Input, Conv2D, GlobalAveragePooling2D, GlobalMaxPooling2D, Dense, Dropout, Flatten, Multiply, MaxPooling2D, Concatenate
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Initial convolutional layer
    conv_layer = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Channel attention path
    avg_pool_path = GlobalAveragePooling2D()(conv_layer)
    avg_pool_path = Dense(units=64, activation='relu')(avg_pool_path)
    avg_pool_path = Dense(units=64, activation='sigmoid')(avg_pool_path)

    max_pool_path = GlobalMaxPooling2D()(conv_layer)
    max_pool_path = Dense(units=64, activation='relu')(max_pool_path)
    max_pool_path = Dense(units=64, activation='sigmoid')(max_pool_path)

    channel_attention_output = Multiply()([avg_pool_path, max_pool_path])
    channel_attention_output = Expand()(-1)(channel_attention_output)

    # Spatial attention path
    avg_pool_path = MaxPooling2D()(conv_layer)
    avg_pool_path = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(avg_pool_path)
    avg_pool_path = Conv2D(filters=1, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='sigmoid')(avg_pool_path)

    max_pool_path = MaxPooling2D()(conv_layer)
    max_pool_path = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(max_pool_path)
    max_pool_path = Conv2D(filters=1, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='sigmoid')(max_pool_path)

    spatial_attention_output = Multiply()([avg_pool_path, max_pool_path])

    # Fuse channel and spatial attention
    fused_features = Multiply()([conv_layer, channel_attention_output, spatial_attention_output])

    # Final layers
    flatten_layer = Flatten()(fused_features)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dropout = Dropout(rate=0.5)(dense1)
    dense2 = Dense(units=10, activation='softmax')(dropout)

    model = Model(inputs=input_layer, outputs=dense2)

    return model