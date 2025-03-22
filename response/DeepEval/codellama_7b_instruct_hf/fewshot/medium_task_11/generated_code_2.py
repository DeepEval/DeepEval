import keras
from keras.layers import Input, Conv2D, GlobalAveragePooling2D, GlobalMaxPooling2D, Dense, Flatten, Activation, Add, Multiply, AveragePooling2D, MaxPooling2D

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Path 1: Global Average Pooling
    conv1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    avg_pool1 = GlobalAveragePooling2D()(conv1)
    dense1 = Dense(units=128, activation='relu')(avg_pool1)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output1 = Dense(units=10, activation='softmax')(dense2)

    # Path 2: Global Max Pooling
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    max_pool2 = GlobalMaxPooling2D()(conv2)
    dense3 = Dense(units=128, activation='relu')(max_pool2)
    dense4 = Dense(units=64, activation='relu')(dense3)
    output2 = Dense(units=10, activation='softmax')(dense4)

    # Channel Attention
    channel_attention = Add()([output1, output2])
    channel_attention = Activation('relu')(channel_attention)
    channel_attention = Multiply()([conv1, channel_attention])

    # Spatial Attention
    spatial_attention = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv2)
    spatial_attention = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(spatial_attention)
    spatial_attention = Activation('relu')(spatial_attention)

    # Fused Feature Map
    fused_feature_map = Multiply()([channel_attention, spatial_attention])
    fused_feature_map = Flatten()(fused_feature_map)

    # Fully Connected Layer
    output = Dense(units=10, activation='softmax')(fused_feature_map)

    model = keras.Model(inputs=input_layer, outputs=output)
    return model