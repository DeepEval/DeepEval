import keras
from keras.layers import Input, Conv2D, DepthwiseConv2D, GlobalAveragePooling2D, Dense, Multiply, Add, Flatten

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # Step 1: Increase dimensionality with a 1x1 convolution
    conv1x1 = Conv2D(filters=9, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Step 2: Extract initial features using a 3x3 depthwise separable convolution
    depthwise_conv = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1x1)
    
    # Step 3: Compute channel attention weights
    global_avg_pooling = GlobalAveragePooling2D()(depthwise_conv)
    fc1 = Dense(units=9 // 2, activation='relu')(global_avg_pooling)  # Reduce to half
    fc2 = Dense(units=9, activation='sigmoid')(fc1)  # Bring back to same number of filters
    attention_weights = fc2
    
    # Step 4: Apply channel attention
    attention_weights_reshaped = keras.layers.Reshape((1, 1, 9))(attention_weights)  # Reshape to match dimensions
    channel_attention = Multiply()([depthwise_conv, attention_weights_reshaped])  # Element-wise multiplication
    
    # Step 5: Reduce dimensionality with a 1x1 convolution
    reduced_conv1x1 = Conv2D(filters=3, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(channel_attention)
    
    # Step 6: Combine with the initial input
    combined_output = Add()([reduced_conv1x1, input_layer])
    
    # Step 7: Pass through flatten layer
    flatten_layer = Flatten()(combined_output)
    
    # Step 8: Fully connected layer for classification
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)
    
    # Build the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model