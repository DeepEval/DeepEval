import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Concatenate, BatchNormalization, Flatten, Dense, Dropout

def dl_model():     

    input_layer = Input(shape=(224, 224, 3))
    
    # Sequential feature extraction layers
    conv1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    avg_pool1 = AveragePooling2D(pool_size=(2, 2), strides=2, padding='same')(conv1)
    
    conv2 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(avg_pool1)
    avg_pool2 = AveragePooling2D(pool_size=(2, 2), strides=2, padding='same')(conv2)
    
    # Additional convolutional layers for feature extraction
    conv3 = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(avg_pool2)
    conv4 = Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv3)
    avg_pool3 = AveragePooling2D(pool_size=(2, 2), strides=2, padding='same')(conv4)
    
    # Flatten the feature maps
    flatten_layer = Flatten()(avg_pool3)
    
    # Two fully connected layers with dropout to mitigate overfitting
    dense1 = Dense(units=1024, activation='relu')(flatten_layer)
    drop1 = Dropout(0.2)(dense1)
    dense2 = Dense(units=512, activation='relu')(drop1)
    drop2 = Dropout(0.2)(dense2)
    
    # Output layer with softmax activation for classification probabilities
    output_layer = Dense(units=1000, activation='softmax')(drop2)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model