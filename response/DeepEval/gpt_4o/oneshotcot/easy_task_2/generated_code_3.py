import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Flatten, Dense, Dropout

def dl_model():
    input_layer = Input(shape=(224, 224, 3))
    
    # First feature extraction block
    conv1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    avg_pool1 = AveragePooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv1)
    
    # Second feature extraction block
    conv2 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(avg_pool1)
    avg_pool2 = AveragePooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv2)
    
    # Additional convolutional layers
    conv3 = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(avg_pool2)
    conv4 = Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv3)
    conv5 = Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv4)
    
    # Final average pooling to reduce dimensionality
    avg_pool3 = AveragePooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv5)
    
    # Flatten the output of the last pooling layer
    flatten_layer = Flatten()(avg_pool3)
    
    # Fully connected layers with dropout
    dense1 = Dense(units=1024, activation='relu')(flatten_layer)
    dropout1 = Dropout(rate=0.5)(dense1)
    
    dense2 = Dense(units=512, activation='relu')(dropout1)
    dropout2 = Dropout(rate=0.5)(dense2)
    
    # Output layer with softmax activation
    output_layer = Dense(units=1000, activation='softmax')(dropout2)
    
    # Build the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model