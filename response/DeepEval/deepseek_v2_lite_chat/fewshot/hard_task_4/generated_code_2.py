import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, GlobalAveragePooling2D, Dense, Lambda, Flatten, Reshape

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Increase channels from 3 to 9 with a 1x1 convolution
    conv1 = Conv2D(filters=9, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Depthwise separable convolution with 3 output channels
    dw_conv1 = Conv2D(filters=3, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', depthwise_init=True)(conv1)
    
    # Global average pooling to compute attention weights
    avg_pooling = GlobalAveragePooling2D()(dw_conv1)
    
    # Two fully connected layers to generate attention weights
    fc1 = Dense(units=128, activation='relu')(avg_pooling)
    fc2 = Dense(units=1, activation='sigmoid')(fc1)  # Sigmoid activation for binary classification (attention weights)
    
    # Reshape attention weights to match the output size of dw_conv1
    attention_weights = Reshape((9, 1))(fc2)  # Assuming 9 channels in dw_conv1
    
    # Element-wise multiplication with the initial features
    weighted_features = keras.backend.batch_dot([attention_weights, dw_conv1], axes=(2, 1))
    
    # Reduce dimensionality with a 1x1 convolution
    conv2 = Conv2D(filters=3, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(weighted_features)
    
    # Flatten and fully connected layers
    flat = Flatten()(conv2)
    dense = Dense(units=10, activation='softmax')(flat)
    
    # Model construction
    model = keras.Model(inputs=input_layer, outputs=dense)
    
    return model

# Check the model structure
model = dl_model()
model.summary()