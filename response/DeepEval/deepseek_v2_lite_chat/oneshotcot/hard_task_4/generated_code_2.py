import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, GlobalAveragePooling2D, Reshape
from keras.models import Model
from keras.layers import multiply

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Increase the number of channels to 3* original (1x1 convolution)
    conv1 = Conv2D(filters=9, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Depthwise separable convolution (3x3)
    depthwise_conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_initializer='he_normal', depthwise_initializer='zeros', activation='relu')(input_layer)
    
    # Global average pooling
    avg_pool = GlobalAveragePooling2D()(depthwise_conv)
    
    # Fully connected layer for channel attention
    fc1 = Dense(units=256, activation='relu')(avg_pool)
    fc2 = Dense(units=128, activation='relu')(fc1)
    
    # Reshape channel attention weights to match the number of channels in initial features
    attention_weights = Reshape((9, 1, 1))(fc2)
    
    # Channel attention: multiply initial features with attention weights
    channel_attention = multiply([depthwise_conv, attention_weights])
    
    # 1x1 convolution to reduce dimensionality
    conv2 = Conv2D(filters=3, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(channel_attention)
    
    # Combine the output with the initial input
    model_output = Concatenate()([conv1, conv2])
    
    # Flatten layer
    flatten = Flatten()(model_output)
    
    # Fully connected layer
    dense = Dense(units=1024, activation='relu')(flatten)
    
    # Output layer
    output_layer = Dense(units=10, activation='softmax')(dense)
    
    # Construct the model
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Create and print the model
model = dl_model()
model.summary()