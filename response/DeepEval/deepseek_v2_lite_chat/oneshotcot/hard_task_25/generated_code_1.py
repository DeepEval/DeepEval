import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, AveragePooling2D, UpSampling2D, Conv2DTranspose

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Main path
    x = Conv2D(filters=64, kernel_size=(1, 1), padding='same')(input_layer)
    x = BatchNormalization()(x)
    x = Conv2D(filters=64, kernel_size=(3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x1 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(x)
    
    # Branch path
    x2 = Conv2D(filters=64, kernel_size=(1, 1), padding='same')(input_layer)
    x2 = BatchNormalization()(x2)
    x2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same')(x2)
    x2 = BatchNormalization()(x2)
    x2 = AveragePooling2D(pool_size=(2, 2), strides=2, padding='same')(x2)
    x2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same')(x2)
    x2 = BatchNormalization()(x2)
    x2 = UpSampling2D()(x2)
    
    # Concatenate and final output
    concat = Concatenate()([x1, x2])
    output = Conv2D(filters=64, kernel_size=(1, 1), padding='same')(concat)
    output = BatchNormalization()(output)
    output = Flatten()(output)
    output = Dense(units=128, activation='relu')(output)
    output = Dense(units=10, activation='softmax')(output)
    
    model = keras.Model(inputs=input_layer, outputs=output)
    
    return model