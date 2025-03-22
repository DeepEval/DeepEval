import keras
from keras.layers import Input, Conv2D, BatchNormalization, Activation, GlobalAveragePooling2D, Dense, Reshape, Concatenate, AveragePooling2D

def dl_model():  
    input_layer = Input(shape=(32, 32, 3))
    
    # Convolutional Layer
    x = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    x = BatchNormalization()(x)

    # Global Average Pooling
    x = GlobalAveragePooling2D()(x)
    
    # Fully Connected Layers
    x = Dense(units=32, activation='relu')(x)
    x = Dense(units=10, activation='relu')(x)

    # Reshape to match initial features
    x = Reshape((32, 32, 32))(x)

    # Multiply with initial features
    x = keras.layers.Multiply()([x, input_layer])

    # Concatenate with input
    x = Concatenate()([input_layer, x])

    # Downsampling
    x = Conv2D(filters=1, kernel_size=(1, 1), strides=(1, 1), padding='same')(x)
    x = AveragePooling2D(pool_size=(2, 2), strides=2, padding='valid')(x)

    # Output Layer
    output_layer = Dense(units=10, activation='softmax')(x) 

    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model