import keras
from keras.layers import Input, Conv2D, DepthwiseConv2D, GlobalAveragePooling2D, Dense, Multiply, Reshape

def dl_model():
    # Input layer
    inputs = Input(shape=(32, 32, 3))
    
    # Increase the dimensionality of the input channels threefold
    x = Conv2D(filters=96, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(inputs)
    
    # 3x3 depthwise separable convolution
    x = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
    
    # Global average pooling
    x = GlobalAveragePooling2D()(x)
    
    # Two fully connected layers for channel attention
    x = Dense(units=128, activation='relu')(x)
    x = Dense(units=128, activation='relu')(x)
    
    # Reshape channel attention weights
    x = Reshape(target_shape=(128,))(x)
    
    # Channel attention weighting
    x = Multiply()([x, x])
    
    # 1x1 convolution for dimensionality reduction
    x = Conv2D(filters=96, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(x)
    
    # Combine attention with initial input
    x = keras.layers.Add()([x, inputs])
    
    # Flatten and fully connected layer for classification
    outputs = Dense(units=10, activation='softmax')(x)
    
    # Model construction
    model = keras.Model(inputs=inputs, outputs=outputs)
    
    return model

# Call the model
model = dl_model()
model.summary()