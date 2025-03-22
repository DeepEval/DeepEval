import keras
from keras.layers import Input, Conv2D, GlobalAveragePooling2D, Dense, Reshape, Multiply, Add, Flatten

def dl_model():  
    
    input_layer = Input(shape=(32, 32, 3))

    # Increase channels threefold
    x = Conv2D(filters=input_layer.shape[-1] * 3, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Initial feature extraction
    x = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', depth_wise=True)(x)

    # Channel attention block
    
    # Global average pooling
    gap = GlobalAveragePooling2D()(x)
    
    # Two fully connected layers for attention weights
    att_weights = Dense(units=gap.shape[-1], activation='relu')(gap)
    att_weights = Dense(units=gap.shape[-1], activation='sigmoid')(att_weights)
    
    # Reshape attention weights to match initial features
    att_weights = Reshape((1, 1, gap.shape[-1]))(att_weights)
    
    # Multiply initial features with attention weights
    x = Multiply()([x, att_weights])
    
    # Reduce dimensionality
    x = Conv2D(filters=input_layer.shape[-1], kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(x)
    
    # Combine with original input
    x = Add()([x, input_layer])
    
    # Flatten and classification
    x = Flatten()(x)
    output_layer = Dense(units=10, activation='softmax')(x)

    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model