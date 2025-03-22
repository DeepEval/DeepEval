import keras
from keras.layers import Input, Conv2D, Dense, GlobalAveragePooling2D, Multiply, Flatten, Add
from keras.layers import DepthwiseConv2D, Reshape

def dl_model():
    # Input layer for CIFAR-10 images, which are 32x32 with 3 channels
    input_layer = Input(shape=(32, 32, 3))
    
    # Step 1: Increase dimensionality of input's channels threefold with a 1x1 convolution
    expanded_channels = Conv2D(filters=9, kernel_size=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Step 2: Extract initial features using a 3x3 depthwise separable convolution
    initial_features = DepthwiseConv2D(kernel_size=(3, 3), padding='same', activation='relu')(expanded_channels)
    
    # Step 3: Compute channel attention weights through global average pooling
    gap = GlobalAveragePooling2D()(initial_features)
    
    # Step 4: Two fully connected layers to generate channel attention weights
    fc1 = Dense(units=4, activation='relu')(gap)
    attention_weights = Dense(units=9, activation='sigmoid')(fc1)
    
    # Step 5: Reshape and multiply the weights with the initial features for channel attention
    attention_weights_reshaped = Reshape((1, 1, 9))(attention_weights)
    attended_features = Multiply()([initial_features, attention_weights_reshaped])
    
    # Step 6: Reduce dimensionality with a 1x1 convolution
    reduced_features = Conv2D(filters=3, kernel_size=(1, 1), padding='same', activation='relu')(attended_features)
    
    # Step 7: Combine the output with the initial input
    combined_output = Add()([reduced_features, input_layer])
    
    # Step 8: Flatten the resulting output
    flatten_layer = Flatten()(combined_output)
    
    # Step 9: Pass through a fully connected layer for classification
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)
    
    # Build the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model