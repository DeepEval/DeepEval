import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, DepthwiseConv2D, GlobalAveragePooling2D, Dense, Multiply, Flatten, Add, Activation
from tensorflow.keras.models import Model

def dl_model():
    # Define the input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Initial 1x1 convolution to increase the number of channels
    conv1 = Conv2D(3, kernel_size=(1, 1), activation='relu')(input_layer)
    
    # Depthwise separable convolution
    depthwise_conv = DepthwiseConv2D(kernel_size=(3, 3), padding='same')(conv1)
    depthwise_conv = Activation('relu')(depthwise_conv)
    
    # Channel attention mechanism
    channel_attention = GlobalAveragePooling2D()(depthwise_conv)
    channel_attention = Dense(depthwise_conv.shape[-1], activation='relu')(channel_attention)
    channel_attention = Dense(depthwise_conv.shape[-1], activation='sigmoid')(channel_attention)
    channel_attention = tf.reshape(channel_attention, (-1, 1, 1, depthwise_conv.shape[-1]))
    channel_attention_applied = Multiply()([depthwise_conv, channel_attention])
    
    # Add the original input with the channel-attention-weighted features
    added = Add()([channel_attention_applied, depthwise_conv])
    
    # Final 1x1 convolution to reduce the number of channels
    conv2 = Conv2D(3, kernel_size=(1, 1), activation='relu')(added)
    
    # Flatten and fully connected layer
    flatten = Flatten()(conv2)
    dense = Dense(10, activation='softmax')(flatten)
    
    # Define the model
    model = Model(inputs=input_layer, outputs=dense)
    
    return model

# Create the model
model = dl_model()
model.summary()