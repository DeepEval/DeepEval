import keras
from keras.layers import Input, Conv2D, DepthwiseConv2D, GlobalAveragePooling2D, Dense, Multiply, Flatten, Add
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # Step 1: Increase the dimensionality of the input's channels threefold with a 1x1 convolution
    conv1 = Conv2D(filters=3, kernel_size=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Step 2: Extract initial features using a 3x3 depthwise separable convolution
    depthwise_conv = DepthwiseConv2D(kernel_size=(3, 3), padding='same', activation='relu')(conv1)
    
    # Step 3: Compute channel attention weights
    gap = GlobalAveragePooling2D()(depthwise_conv)
    dense1 = Dense(units=depthwise_conv.shape[-1]//4, activation='relu')(gap)
    dense2 = Dense(units=depthwise_conv.shape[-1], activation='sigmoid')(dense1)
    reshaped_weights = Dense(units=depthwise_conv.shape[-1], activation='sigmoid')(dense2)
    
    # Step 4: Multiply the weights with the initial features to achieve channel attention weighting
    scaled_features = Multiply()([reshaped_weights[:, :, :, None], depthwise_conv])
    
    # Step 5: A 1x1 convolution reduces the dimensionality
    conv2 = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(scaled_features)
    
    # Step 6: Combine the output with the initial input
    added = Add()([conv2, depthwise_conv])
    
    # Step 7: Flatten the result and pass through a fully connected layer for classification
    flattened = Flatten()(added)
    output_layer = Dense(units=10, activation='softmax')(flattened)
    
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model