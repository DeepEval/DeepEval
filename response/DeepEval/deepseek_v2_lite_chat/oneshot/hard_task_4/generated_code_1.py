import keras
from keras.layers import Input, Conv2D, DepthwiseConv2D, GlobalAveragePooling2D, Dense, Concatenate, multiply, Add

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Conv layer to increase channels from 3 to 9
    conv1 = Conv2D(filters=9, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='sigmoid')(input_layer)
    
    # Depthwise separable convolution
    dw_conv = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', name='depthwise_conv')(conv1)
    
    # Global average pooling
    avg_pool = GlobalAveragePooling2D()(dw_conv)
    
    # Fully connected layer for channel attention
    fc1 = Dense(units=128, activation='relu')(avg_pool)
    fc2 = Dense(units=64, activation='relu')(fc1)
    
    # Reshape FC output to match dw_conv_output's channels
    attention_weights = Dense(kernel_size=(1, 1), activation='sigmoid')(fc2)
    attention_weights = tf.expand_dims(attention_weights, axis=[2])  # Add channel dimension
    attention_weights = tf.tile(attention_weights, [1, 1, dw_conv.shape[2], 1])  # Tile weights across channels
    
    # Element-wise multiplication with dw_conv_output
    attention_output = multiply([dw_conv, attention_weights])
    
    # Reduce dimensionality with 1x1 convolution
    conv2 = Conv2D(filters=9, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='sigmoid')(attention_output)
    
    # Add the 1x1 conv output with the original conv1 output
    combined_output = Add()([conv1, conv2])
    
    # Flatten and fully connected layers for classification
    flatten = Flatten()(combined_output)
    dense1 = Dense(units=128, activation='relu')(flatten)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    # Create the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Instantiate and print the model
model = dl_model()
model.summary()