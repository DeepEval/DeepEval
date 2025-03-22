import keras
from keras.layers import Input, Conv2D, GlobalAveragePooling2D, GlobalMaxPooling2D, Dense, Concatenate, BatchNormalization, Add

def dl_model():     

    input_layer = Input(shape=(32, 32, 3))
    
    # Block 1: Channel Attention
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    path1 = GlobalAveragePooling2D()(conv1)
    path1 = Dense(units=128, activation='relu')(path1)
    path1 = Dense(units=3, activation='sigmoid')(path1)  # Output channels match input

    path2 = GlobalMaxPooling2D()(conv1)
    path2 = Dense(units=128, activation='relu')(path2)
    path2 = Dense(units=3, activation='sigmoid')(path2)

    attention_output = Add()([path1, path2])
    attention_output = keras.layers.multiply([conv1, attention_output]) 

    # Block 2: Spatial Feature Extraction
    avg_pool = GlobalAveragePooling2D()(attention_output)
    max_pool = GlobalMaxPooling2D()(attention_output)
    spatial_features = Concatenate()([avg_pool, max_pool])
    spatial_features = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='sigmoid')(spatial_features)
    spatial_features = keras.layers.multiply([attention_output, spatial_features])

    # Additional branch for output channel alignment
    branch_conv = Conv2D(filters=3, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(spatial_features)

    # Final Combine
    output = Add()([spatial_features, branch_conv])
    output = keras.layers.Activation('relu')(output)  

    output_layer = Dense(units=10, activation='softmax')(output)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model