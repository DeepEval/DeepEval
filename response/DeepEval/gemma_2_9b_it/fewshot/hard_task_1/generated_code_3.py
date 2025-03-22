import keras
from keras.layers import Input, Conv2D, GlobalAveragePooling2D, GlobalMaxPooling2D, Flatten, Dense, Add, Activation, Concatenate
    
def dl_model():

    input_layer = Input(shape=(32, 32, 3))
    
    # Block 1: Channel Attention
    
    conv_initial = Conv2D(filters=3, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)

    path1 = GlobalAveragePooling2D()(conv_initial)
    path1 = Dense(units=128, activation='relu')(path1)
    path1 = Dense(units=3, activation='sigmoid')(path1) 

    path2 = GlobalMaxPooling2D()(conv_initial)
    path2 = Dense(units=128, activation='relu')(path2)
    path2 = Dense(units=3, activation='sigmoid')(path2)

    channel_attention = Add()([path1, path2]) 
    channel_attention = Activation('sigmoid')(channel_attention)

    features_with_attention = input_layer * channel_attention
    
    # Block 2: Spatial Feature Extraction
    
    avg_pool = GlobalAveragePooling2D()(features_with_attention)
    max_pool = GlobalMaxPooling2D()(features_with_attention)
    spatial_features = Concatenate()([avg_pool, max_pool])
    spatial_features = Conv2D(filters=3, kernel_size=(1, 1), strides=(1, 1), padding='same')(spatial_features)
    spatial_features = Activation('sigmoid')(spatial_features)
    
    features_combined = features_with_attention * spatial_features 

    # Final Branch and Classification
    
    final_branch = Conv2D(filters=3, kernel_size=(1, 1), strides=(1, 1), padding='same')(features_combined)
    output = Add()([final_branch, features_combined])
    output = Activation('relu')(output) 
    output = Flatten()(output)
    output_layer = Dense(units=10, activation='softmax')(output)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model