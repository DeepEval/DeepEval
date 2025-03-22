import keras
from keras.layers import Input, Conv2D, DepthwiseConv2D, GlobalAveragePooling2D, Dense, concatenate, multiply

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Increase channels from 3 to 9 with 1x1 convolution
    conv_dim_increase = Conv2D(filters=9, kernel_size=(1, 1), padding='same')(input_layer)
    
    # Depthwise separable convolution for initial feature extraction
    depthwise_conv = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same')(conv_dim_increase)
    
    # Global average pooling to compute channel attention weights
    avg_pool = GlobalAveragePooling2D()(depthwise_conv)
    
    # Fully connected layer for channel attention
    fc1 = Dense(units=128, activation='relu')(avg_pool)
    fc2 = Dense(units=64, activation='relu')(fc1)
    
    # Reshape channel attention weights to match the number of channels in the initial features
    attention_weights = Dense(units=9 * 16)(fc2)  # Adjust the units and number of channels as needed
    attention_weights = tf.reshape(attention_weights, (-1, 9, 16))  # Reshape to match the initial features
    
    # Multiply attention weights with initial features
    attention_enhanced = multiply([depthwise_conv, attention_weights])
    
    # Reduce dimensionality with 1x1 convolution
    reduced_features = Conv2D(filters=9, kernel_size=(1, 1), padding='same')(attention_enhanced)
    
    # Flatten the output for fully connected layers
    flattened = Flatten()(reduced_features)
    
    # Fully connected layer for final classification
    dense = Dense(units=10, activation='softmax')(flattened)
    
    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=dense)
    
    return model