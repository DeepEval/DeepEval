import keras
from keras.layers import Input, Conv2D, DepthwiseConv2D, GlobalAveragePooling2D, Dense, Reshape, Multiply, Add, Flatten, Dense

def dl_model():
    
    input_layer = Input(shape=(32, 32, 3))
    
    # Increasing dimensionality of input's channels threefold with a 1x1 convolution
    conv1 = Conv2D(filters=3*3, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Extracting initial features using a 3x3 depthwise separable convolution
    conv2 = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
    
    # Computing channel attention weights
    global_avg_pool = GlobalAveragePooling2D()(conv2)
    dense1 = Dense(units=3*3, activation='relu')(global_avg_pool)
    dense2 = Dense(units=3*3, activation='sigmoid')(dense1)
    weights = Reshape(target_shape=(3, 3))(dense2)
    
    # Applying channel attention weighting
    weighted_features = Multiply()([conv2, weights])
    
    # Reducing dimensionality and combining with initial input
    reduced_features = Conv2D(filters=3, kernel_size=(1, 1), strides=(1, 1), padding='same')(weighted_features)
    combined_features = Add()([reduced_features, input_layer])
    
    # Flattening and classification
    flatten_layer = Flatten()(combined_features)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model