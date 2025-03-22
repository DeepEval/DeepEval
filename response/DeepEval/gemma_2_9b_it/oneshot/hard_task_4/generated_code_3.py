import keras
from keras.layers import Input, Conv2D, GlobalAveragePooling2D, Dense, Reshape, Multiply, Add, Flatten

def dl_model():     
    input_layer = Input(shape=(32, 32, 3))

    # Increase channels threefold
    x = Conv2D(filters=9, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Initial feature extraction with depthwise separable convolution
    x = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', depth_wise=True)(x) 

    # Channel Attention
    features = x
    
    # Global average pooling
    pool_output = GlobalAveragePooling2D()(x) 

    # Two fully connected layers for attention weights
    att_weights = Dense(units=64, activation='relu')(pool_output)
    att_weights = Dense(units=16, activation='sigmoid')(att_weights)  

    # Reshape attention weights
    att_weights = Reshape((32, 32, 16))(att_weights)

    # Multiply features with attention weights
    x = Multiply()([features, att_weights])
    
    # Reduce dimensionality with 1x1 convolution
    x = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(x)

    # Combine with initial input
    x = Add()([x, input_layer])

    # Flatten and final classification
    x = Flatten()(x)
    output_layer = Dense(units=10, activation='softmax')(x)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model