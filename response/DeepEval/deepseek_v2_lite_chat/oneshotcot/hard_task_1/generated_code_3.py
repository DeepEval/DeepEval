import keras
from keras.layers import Input, Conv2D, MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D, GlobalMaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Activation

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Block 1: Feature Extraction
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(input_layer)
    avg_pool1 = AveragePooling2D(pool_size=(2, 2))(conv1)
    
    # Feature Extraction paths
    path1 = GlobalAveragePooling2D()(avg_pool1)
    path1_dense = Dense(units=128, activation='relu')(path1)
    
    path2 = GlobalMaxPooling2D()(avg_pool1)
    path2_dense = Dense(units=128, activation='relu')(path2)
    
    # Concatenate and generate channel attention weights
    concat = Concatenate()([path1_dense, path2_dense])
    channel_attention = Dense(units=1)(concat)
    channel_attention = Activation('sigmoid')(channel_attention)
    
    # Element-wise multiplication with original features
    output1 = channel_attention * conv1
    
    # Block 2: Spatial Feature Extraction
    avg_pool2 = AveragePooling2D(pool_size=(3, 3))(output1)
    max_pool2 = MaxPooling2D(pool_size=(1, 1))(output1)
    
    avg_pool2_flat = Flatten()(avg_pool2)
    max_pool2_flat = Flatten()(max_pool2)
    
    concat_spatial = Concatenate()(outputs=[avg_pool2_flat, max_pool2_flat])
    spatial_attention = Dense(units=1)(concat_spatial)
    spatial_attention = Activation('sigmoid')(spatial_attention)
    
    output2 = spatial_attention * avg_pool2
    output3 = spatial_attention * max_pool2
    
    # Final branch for output channels alignment
    conv3 = Conv2D(filters=3, kernel_size=(1, 1), padding='same', activation='relu')(output1)
    add = Add()([output1, conv3])
    final_output = Activation('softmax')(add)
    
    # Model construction
    model = keras.Model(inputs=input_layer, outputs=[final_output, output2, output3])
    
    return model

# Example usage:
# model = dl_model()
# model.summary()