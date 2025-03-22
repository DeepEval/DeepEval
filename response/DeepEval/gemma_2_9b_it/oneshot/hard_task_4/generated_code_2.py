import keras
from keras.layers import Input, Conv2D, GlobalAveragePooling2D, Dense, Reshape, Multiply, Add, Flatten

def dl_model():  
    
    input_layer = Input(shape=(32, 32, 3))
    
    # Increase channels threefold
    x = Conv2D(filters=input_layer.shape[-1] * 3, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Depthwise separable convolution for initial feature extraction
    x = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', depth_wise=True)(x)

    # Channel attention block
    features = x
    pool = GlobalAveragePooling2D()(x)
    
    attention_weights = Dense(units=features.shape[-1], activation='sigmoid')(pool)
    attention_weights = Reshape((1, 1, features.shape[-1]))(attention_weights)

    x = Multiply()([features, attention_weights])

    x = Conv2D(filters=input_layer.shape[-1], kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(x)

    # Add initial input with the output of the channel attention block
    x = Add()([input_layer, x])

    # Flatten and dense layer for classification
    x = Flatten()(x)
    x = Dense(units=10, activation='softmax')(x)

    model = keras.Model(inputs=input_layer, outputs=x)

    return model