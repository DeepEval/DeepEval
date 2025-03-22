import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout

def dl_model():
    input_layer = Input(shape=(28, 28, 1))
    
    # Average Pooling with 5x5 window and 3x3 stride
    conv = Conv2D(filters=32, kernel_size=(5, 5), strides=(3, 3), padding='same', activation='relu')(input_layer)
    pool = MaxPooling2D(pool_size=(2, 2))(conv)
    
    # 1x1 Convolution
    conv_1 = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(pool)
    
    # Flattening
    flatten = Flatten()(conv_1)
    
    # Fully Connected Layers
    dense1 = Dense(units=128, activation='relu')(flatten)
    dropout = Dropout(rate=0.5)(dense1)
    dense2 = Dense(units=64, activation='relu')(dropout)
    
    # Output Layer
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model