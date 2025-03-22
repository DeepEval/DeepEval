import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(28, 28, 1))
    
    # First block: Convolution, Convolution, Max Pooling
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(input_layer)
    conv2 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv2)
    
    # Second block: Convolution, Convolution, Max Pooling
    conv3 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(pool1)
    conv4 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(conv3)
    pool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv4)
    
    # Third block: Convolution, Convolution, Convolution, Max Pooling
    conv5 = Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same')(pool2)
    conv6 = Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same')(conv5)
    conv7 = Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same')(conv6)
    pool3 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv7)
    
    # Fourth block: Convolution, Convolution, Convolution, Max Pooling
    conv8 = Conv2D(filters=256, kernel_size=(3, 3), activation='relu', padding='same')(pool3)
    conv9 = Conv2D(filters=256, kernel_size=(3, 3), activation='relu', padding='same')(conv8)
    conv10 = Conv2D(filters=256, kernel_size=(3, 3), activation='relu', padding='same')(conv9)
    pool4 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv10)
    
    # Flatten and Fully Connected Layers
    flatten = Flatten()(pool4)
    dense1 = Dense(units=128, activation='relu')(flatten)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    # Create the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model