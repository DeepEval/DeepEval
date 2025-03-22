import keras
from keras.layers import Input, Conv2D, SeparableConv2D, GlobalAveragePooling2D, Reshape, Multiply, Flatten, Dense, Add

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # Increase the dimensionality of the input's channels threefold with a 1x1 convolution
    expanded_channels = Conv2D(filters=9, kernel_size=(1, 1), activation='relu', padding='same')(input_layer)
    
    # Extract initial features using a 3x3 depthwise separable convolution
    initial_features = SeparableConv2D(filters=9, kernel_size=(3, 3), activation='relu', padding='same')(expanded_channels)
    
    # Compute channel attention weights
    gap = GlobalAveragePooling2D()(initial_features)
    
    # Use two fully connected layers to generate weights
    fc1 = Dense(units=9//2, activation='relu')(gap)
    fc2 = Dense(units=9, activation='sigmoid')(fc1)
    
    # Reshape and apply channel attention weights
    channel_attention_weights = Reshape((1, 1, 9))(fc2)
    attention_applied_features = Multiply()([initial_features, channel_attention_weights])
    
    # Reduce dimensionality with a 1x1 convolution
    reduced_features = Conv2D(filters=3, kernel_size=(1, 1), activation='relu', padding='same')(attention_applied_features)
    
    # Combine with the initial input
    combined_output = Add()([input_layer, reduced_features])
    
    # Flatten and fully connected layer for classification
    flatten_layer = Flatten()(combined_output)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model