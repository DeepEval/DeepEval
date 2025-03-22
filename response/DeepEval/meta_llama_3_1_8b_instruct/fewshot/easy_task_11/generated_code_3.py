import keras
from keras.layers import Input, AveragePooling2D, Conv2D, Flatten, Dense, Dropout

def dl_model():
    
    input_layer = Input(shape=(28, 28, 1))
    
    # Apply average pooling layer with a 5x5 window and a 3x3 stride
    avg_pooling = AveragePooling2D(pool_size=(5, 5), strides=(3, 3), padding='same')(input_layer)
    
    # Apply 1x1 convolutional layer to enhance depth and introduce nonlinearity
    conv = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(avg_pooling)
    
    # Flatten the feature maps
    flatten_layer = Flatten()(conv)
    
    # Apply two fully connected layers with dropout to mitigate overfitting
    dense1 = Dense(units=64, activation='relu')(flatten_layer)
    dropout = Dropout(0.2)(dense1)
    dense2 = Dense(units=10, activation='softmax')(dropout)
    
    model = keras.Model(inputs=input_layer, outputs=dense2)
    
    return model