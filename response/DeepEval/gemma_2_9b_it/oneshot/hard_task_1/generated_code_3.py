import keras
from keras.layers import Input, Conv2D, GlobalAveragePooling2D, GlobalMaxPooling2D, Dense, Concatenate, BatchNormalization, Add, Lambda
from keras.utils import  plot_model

def dl_model():
    
    input_layer = Input(shape=(32, 32, 3))
    
    # Initial convolutional layer
    conv_layer = Conv2D(filters=3, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Block 1: Channel Attention
    
    # Path 1: Global average pooling
    gap_output = GlobalAveragePooling2D()(conv_layer)
    dense1_path1 = Dense(units=128, activation='relu')(gap_output)
    dense2_path1 = Dense(units=3, activation='sigmoid')(dense1_path1)  

    # Path 2: Global max pooling
    gmp_output = GlobalMaxPooling2D()(conv_layer)
    dense1_path2 = Dense(units=128, activation='relu')(gmp_output)
    dense2_path2 = Dense(units=3, activation='sigmoid')(dense1_path2)  

    # Concatenate outputs and apply sigmoid
    channel_attention = Add()([dense2_path1, dense2_path2]) 
    channel_attention = Lambda(lambda x: keras.backend.expand_dims(x, axis=1))(channel_attention) 
    channel_attention = Lambda(lambda x: keras.backend.expand_dims(x, axis=2))(channel_attention)
    
    # Element-wise multiplication for channel weighting
    weighted_features = Multiply()([conv_layer, channel_attention])

    # Block 2: Spatial Feature Extraction
    avg_pool = AveragePooling2D(pool_size=(2, 2), strides=2, padding='same')(weighted_features)
    max_pool = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(weighted_features)
    spatial_features = Concatenate(axis=3)([avg_pool, max_pool])
    spatial_features = Conv2D(filters=3, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='sigmoid')(spatial_features)

    # Combine spatial and channel features
    combined_features = Add()([weighted_features, spatial_features]) 

    # Final branch
    final_branch = Conv2D(filters=3, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(combined_features)

    # Classification layer
    output_layer = Dense(units=10, activation='softmax')(Flatten()(final_branch))

    model = keras.Model(inputs=input_layer, outputs=output_layer)
    return model