import keras
from keras.layers import Input, Conv2D, AveragePooling2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, GlobalAveragePooling2D, GlobalMaxPooling2D, Multiply
from keras.regularizers import l2

def dl_model():     

    input_layer = Input(shape=(32, 32, 3))

    # Convolutional layer
    conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Block 1: Channel attention
    def block1(input_tensor):
        path1 = GlobalAveragePooling2D()(input_tensor)
        path1 = Dense(units=64, activation='relu')(path1)
        path1 = Dense(units=3, activation='relu')(path1)

        path2 = GlobalMaxPooling2D()(input_tensor)
        path2 = Dense(units=64, activation='relu')(path2)
        path2 = Dense(units=3, activation='relu')(path2)

        output_tensor = Concatenate()([path1, path2])
        output_tensor = Dense(units=3, activation='sigmoid')(output_tensor)
        return output_tensor
    
    block1_output = block1(conv)
    channel_attention = Multiply()([block1_output, conv])
    channel_attention = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(channel_attention)

    # Block 2: Spatial attention
    def block2(input_tensor):
        avg_pool = AveragePooling2D(pool_size=(2, 2), strides=2, padding='same')(input_tensor)
        max_pool = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(input_tensor)
        output_tensor = Concatenate()([avg_pool, max_pool])
        output_tensor = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='sigmoid')(output_tensor)
        return output_tensor

    spatial_attention = block2(channel_attention)
    channel_attention = Multiply()([channel_attention, spatial_attention])

    # Ensure output channels align with input channels
    ensure_channels = Conv2D(filters=3, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(channel_attention)

    # Add attention features to main path
    final_features = Add()([channel_attention, ensure_channels])

    # Final fully connected layer
    flatten_layer = Flatten()(final_features)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model