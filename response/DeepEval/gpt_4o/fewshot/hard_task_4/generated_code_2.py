import keras
from keras.layers import Input, Conv2D, DepthwiseConv2D, GlobalAveragePooling2D, Dense, Multiply, Flatten, Add

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # Step 1: Increase the dimensionality threefold using a 1x1 convolution
    expanded_channels = Conv2D(filters=9, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Step 2: Extract initial features using a 3x3 depthwise separable convolution
    initial_features = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(expanded_channels)
    
    # Step 3: Channel attention mechanism
    gap = GlobalAveragePooling2D()(initial_features)
    
    dense1 = Dense(units=9 // 2, activation='relu')(gap)  # Reduce the dimensionality
    dense2 = Dense(units=9, activation='sigmoid')(dense1)  # Output the attention weights
    
    # Reshape the attention weights
    attention_weights = keras.layers.Reshape((1, 1, 9))(dense2)
    
    # Multiply initial features with the attention weights
    attention_features = Multiply()([initial_features, attention_weights])
    
    # Step 4: Reduce dimensionality using a 1x1 convolution
    reduced_features = Conv2D(filters=3, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(attention_features)
    
    # Step 5: Combine with the initial input
    combined_features = Add()([input_layer, reduced_features])
    
    # Step 6: Flatten and classify
    flatten_layer = Flatten()(combined_features)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model