import keras
from keras.layers import Input, AveragePooling2D, Conv2D, Flatten, Dense, Dropout

def dl_model():
    
    input_layer = Input(shape=(28, 28, 1))
    
    # Average pooling layer
    avg_pool = AveragePooling2D(pool_size=(5, 5), strides=(3, 3), padding='valid')(input_layer)
    
    # 1x1 convolutional layer
    conv = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(avg_pool)
    
    # Flatten layer
    flatten_layer = Flatten()(conv)
    
    # Dropout layer to prevent overfitting
    dropout = Dropout(0.2)(flatten_layer)
    
    # Two fully connected layers
    dense1 = Dense(units=128, activation='relu')(dropout)
    dense2 = Dense(units=10, activation='softmax')(dense1)
    
    # Create the model
    model = keras.Model(inputs=input_layer, outputs=dense2)
    
    return model