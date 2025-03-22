import keras
from keras.layers import Input, Conv2D, GlobalAveragePooling2D, GlobalMaxPooling2D, Dense, concatenate, Activation, AveragePooling2D, MaxPooling2D

def dl_model():     

        input_layer = Input(shape=(32, 32, 3))
        conv_layer = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(input_layer)

        # Global Average Pooling Path
        avg_pool_path = GlobalAveragePooling2D()(conv_layer)
        avg_fc1 = Dense(units=128, activation='relu')(avg_pool_path)
        avg_fc2 = Dense(units=64, activation='relu')(avg_fc1)

        # Global Max Pooling Path
        max_pool_path = GlobalMaxPooling2D()(conv_layer)
        max_fc1 = Dense(units=128, activation='relu')(max_pool_path)
        max_fc2 = Dense(units=64, activation='relu')(max_fc1)

        # Channel Attention
        channel_attention = concatenate([avg_fc2, max_fc2])
        channel_attention = Activation('sigmoid')(channel_attention)

        channel_weighted_features = conv_layer * channel_attention

        # Spatial Feature Extraction
        avg_pool = AveragePooling2D(pool_size=(2, 2))(channel_weighted_features)
        max_pool = MaxPooling2D(pool_size=(2, 2))(channel_weighted_features)
        spatial_features = concatenate([avg_pool, max_pool], axis=3)

        # Concatenate Channel and Spatial Features
        fused_features = concatenate([spatial_features, channel_weighted_features], axis=3)

        # Flatten and Classify
        flatten_layer = Flatten()(fused_features)
        output_layer = Dense(units=10, activation='softmax')(flatten_layer)

        model = keras.Model(inputs=input_layer, outputs=output_layer)

        return model