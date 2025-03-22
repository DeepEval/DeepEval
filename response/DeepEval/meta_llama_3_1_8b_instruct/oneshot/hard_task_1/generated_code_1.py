import keras
from keras.layers import Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, GlobalMaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense
from keras.layers import Multiply, Add
from keras.initializers import RandomNormal

def dl_model():
    
    input_layer = Input(shape=(32, 32, 3))

    # Block 1: Channel Attention
    def block1(input_tensor):
        
        path1 = GlobalAveragePooling2D()(input_tensor)
        path1 = Dense(units=64, activation='relu')(path1)
        path1 = Dense(units=3, activation='sigmoid')(path1)
        
        path2 = GlobalMaxPooling2D()(input_tensor)
        path2 = Dense(units=64, activation='relu')(path2)
        path2 = Dense(units=3, activation='sigmoid')(path2)
        
        channel_attention_weights = Add()([path1, path2])
        channel_attention_weights = Dense(units=3, activation='sigmoid')(channel_attention_weights)
        
        channel_attention = Multiply()([input_tensor, channel_attention_weights])
        return channel_attention
    
    # Block 2: Spatial Features
    def block2(input_tensor):
        
        avg_pool = AveragePooling2D(pool_size=(2, 2))(input_tensor)
        max_pool = MaxPooling2D(pool_size=(2, 2))(input_tensor)
        spatial_features = Concatenate()([avg_pool, max_pool])
        spatial_features = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='sigmoid')(spatial_features)
        
        normalized_features = Multiply()([spatial_features, block1(input_tensor)])
        return normalized_features
    
    # Block 3: Final Path
    block1_output = block1(input_layer)
    block2_output = block2(block1_output)
    block2_output = Conv2D(filters=3, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='sigmoid')(block2_output)
    
    final_path = Add()([block2_output, block1_output])
    final_path = Flatten()(final_path)
    output_layer = Dense(units=10, activation='softmax')(final_path)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model