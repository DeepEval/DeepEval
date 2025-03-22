import keras
from keras.models import Model
from keras.layers import Input, Conv2D, Add, ZeroPadding2D, MaxPooling2D, AveragePooling2D, Concatenate, Flatten, Dense

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Convolutional layers
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(input_layer)
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(conv1)
    conv3 = Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same')(conv2)
    
    # Add first two convolutional layers to the third
    add_layer = Add()([conv3, conv2])
    
    # Separate convolutional layer for input
    sep_conv = Conv2D(filters=64, kernel_size=(1, 1), activation='relu', padding='same')(input_layer)
    
    # Pooling layers
    pool1 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(conv3)
    pool2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(add_layer)
    
    # Flatten and connect both paths
    flat = Flatten()(Concatenate()([pool1, pool2]))
    
    # Fully connected layers
    dense1 = Dense(units=512, activation='relu')(flat)
    dense2 = Dense(units=10, activation='softmax')(dense1)
    
    # Model construction
    model = Model(inputs=input_layer, outputs=dense2)
    
    return model

# Instantiate and return the model
model = dl_model()
model.summary()