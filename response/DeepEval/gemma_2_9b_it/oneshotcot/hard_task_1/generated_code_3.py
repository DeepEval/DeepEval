import keras
from keras.layers import Input, Conv2D, GlobalAveragePooling2D, GlobalMaxPooling2D, Flatten, Dense, Concatenate, Add, Multiply, Activation

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Initial convolutional layer
    conv_layer = Conv2D(filters=3, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Block 1: Channel Attention
    path1 = GlobalAveragePooling2D()(conv_layer)
    path1 = Dense(units=64, activation='relu')(path1)
    path1 = Dense(units=3, activation='sigmoid')(path1)  

    path2 = GlobalMaxPooling2D()(conv_layer)
    path2 = Dense(units=64, activation='relu')(path2)
    path2 = Dense(units=3, activation='sigmoid')(path2)

    channel_attention = Add()([path1, path2])
    channel_attention = Multiply()([channel_attention, conv_layer])

    # Block 2: Spatial Feature Extraction
    avg_pool = GlobalAveragePooling2D()(channel_attention)
    max_pool = GlobalMaxPooling2D()(channel_attention)
    spatial_features = Concatenate()([avg_pool, max_pool])
    spatial_features = Conv2D(filters=3, kernel_size=(1, 1))(spatial_features)
    spatial_features = Activation('sigmoid')(spatial_features)
    
    # Combine channel and spatial features
    combined_features = Multiply()([channel_attention, spatial_features])
    
    # Additional branch
    branch_layer = Conv2D(filters=3, kernel_size=(1, 1))(combined_features)

    # Final addition and activation
    output = Add()([combined_features, branch_layer])
    output = Activation('relu')(output)

    # Classification layer
    output_layer = Flatten()(output)
    output_layer = Dense(units=10, activation='softmax')(output_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)
    return model