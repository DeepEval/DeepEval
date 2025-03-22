import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Concatenate

def dl_model():
    # Input layer
    input_layer = Input(shape=(28, 28, 1))
    
    # Block 1: Convolutional layers and max pooling
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(input_layer)
    pool1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv1)
    
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(pool1)
    pool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv2)
    
    conv3 = Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same')(pool2)
    pool3 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv3)
    
    # Flatten and feed to fully connected layers
    flat = Flatten()(pool3)
    
    # Fully connected layers
    dense1 = Dense(units=1024, activation='relu')(flat)
    dense2 = Dense(units=512, activation='relu')(dense1)
    output = Dense(units=10, activation='softmax')(dense2)
    
    # Model
    model = keras.Model(inputs=input_layer, outputs=output)
    
    return model