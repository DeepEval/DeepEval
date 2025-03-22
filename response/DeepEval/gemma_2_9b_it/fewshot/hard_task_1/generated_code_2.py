import keras
from keras.layers import Input, Conv2D, GlobalAveragePooling2D, GlobalMaxPooling2D, Dense, Concatenate, Add, Activation, Reshape

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # Block 1: Channel Attention
    conv_1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    path1 = GlobalAveragePooling2D()(conv_1)
    path1 = Dense(units=128, activation='relu')(path1)
    path1 = Dense(units=32, activation='relu')(path1)
    
    path2 = GlobalMaxPooling2D()(conv_1)
    path2 = Dense(units=128, activation='relu')(path2)
    path2 = Dense(units=32, activation='relu')(path2)
    
    channel_attention = Add()([path1, path2])
    channel_attention = Activation('sigmoid')(channel_attention)
    channel_attention = Reshape(target_shape=(32, 32, 1))(channel_attention)
    output_conv_1 = conv_1 * channel_attention
    
    # Block 2: Spatial Feature Extraction
    avg_pool = GlobalAveragePooling2D()(output_conv_1)
    max_pool = GlobalMaxPooling2D()(output_conv_1)
    spatial_features = Concatenate(axis=3)([avg_pool, max_pool])
    spatial_features = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='sigmoid')(spatial_features)
    
    output_block2 = spatial_features * output_conv_1
    
    # Final Branch
    final_branch = Conv2D(filters=3, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(output_block2)
    
    # Output Layer
    output_layer = Conv2D(filters=10, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='softmax')(final_branch)

    model = keras.Model(inputs=input_layer, outputs=output_layer)
    return model