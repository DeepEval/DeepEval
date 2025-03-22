import keras
from keras.layers import Input, Conv2D, DepthwiseConv2D, GlobalAveragePooling2D, Dense, Multiply, Flatten, Add
from keras.models import Model
import tensorflow as tf

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # Initial 1x1 convolution to increase channels
    x = Conv2D(filters=3, kernel_size=(1, 1), padding='same', activation='relu')(input_layer)
    
    # 3x3 depthwise separable convolution
    x = DepthwiseConv2D(kernel_size=(3, 3), padding='same', activation='relu')(x)
    
    # Channel attention mechanism
    channel_avg_pool = GlobalAveragePooling2D()(x)
    dense1 = Dense(units=x.shape[-1]//4, activation='relu')(channel_avg_pool)
    dense2 = Dense(units=x.shape[-1], activation='sigmoid')(dense1)
    dense2_reshaped = tf.reshape(dense2, (1, 1, -1))
    
    # Apply weights to the initial features
    x_scaled = Multiply()([x, dense2_reshaped])
    
    # 1x1 convolution to reduce dimensionality
    x_reduced = Conv2D(filters=3, kernel_size=(1, 1), padding='same', activation='relu')(x_scaled)
    
    # Add the initial input with the scaled features
    x_final = Add()([x_reduced, input_layer])
    
    # Flatten and fully connected layer
    flatten_layer = Flatten()(x_final)
    dense3 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense3)
    
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model