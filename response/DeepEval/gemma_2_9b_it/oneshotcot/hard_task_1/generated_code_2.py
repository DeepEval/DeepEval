import keras
from keras.layers import Input, Conv2D, GlobalAveragePooling2D, GlobalMaxPooling2D, Dense, Concatenate, BatchNormalization, Add

def dl_model():     

    input_layer = Input(shape=(32, 32, 3))
    
    # Initial Convolutional Layer
    conv1 = Conv2D(filters=3, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer) 

    # Block 1: Channel Attention
    avg_pool = GlobalAveragePooling2D()(conv1)
    avg_fc1 = Dense(units=128, activation='relu')(avg_pool)
    avg_fc2 = Dense(units=3, activation='sigmoid')(avg_fc1)  

    max_pool = GlobalMaxPooling2D()(conv1)
    max_fc1 = Dense(units=128, activation='relu')(max_pool)
    max_fc2 = Dense(units=3, activation='sigmoid')(max_fc1)  

    channel_attention = Add()([avg_fc2, max_fc2])

    # Apply channel attention weights
    att_features = conv1 * channel_attention  

    # Block 2: Spatial Feature Extraction
    avg_pool2 = AveragePooling2D(pool_size=(2, 2), strides=2, padding='same')(att_features)
    max_pool2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(att_features)
    
    concat_features = Concatenate(axis=3)([avg_pool2, max_pool2])
    conv2 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='sigmoid')(concat_features)

    # Element-wise Multiplication
    spatial_features = conv2 * att_features

    # Additional Branch
    branch_conv = Conv2D(filters=3, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(spatial_features)
    
    # Combine branches
    output = Add()([spatial_features, branch_conv])

    # Final Classification
    output = Flatten()(output)
    output_layer = Dense(units=10, activation='softmax')(output) 

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model