import keras
from keras.layers import Input, Conv2D, Add, MaxPooling2D, AveragePooling2D, Flatten, Dense

def dl_model():
    # Input Layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Convolutional Layer 1
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(input_layer)
    
    # Convolutional Layer 2
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(conv1)
    
    # Convolutional Layer 3
    conv3 = Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same')(conv2)
    
    # Average Pooling on Convolutional Layer 3
    avg_pool = AveragePooling2D(pool_size=(2, 2))(conv3)
    
    # Separate Convolutional Layer processing input directly
    direct_conv = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(input_layer)
    
    # Add paths together
    add_layer = Add()([avg_pool, direct_conv])
    
    # Flatten and pass through Fully Connected Layers
    flatten = Flatten()(add_layer)
    dense1 = Dense(units=128, activation='relu')(flatten)
    dense2 = Dense(units=10, activation='softmax')(dense1)
    
    model = keras.Model(inputs=input_layer, outputs=dense2)
    
    return model

model = dl_model()
model.summary()