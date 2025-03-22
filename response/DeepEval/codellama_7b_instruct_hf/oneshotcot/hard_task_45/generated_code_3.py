import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # First block
    branch1 = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same')(input_layer)
    branch1 = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same')(branch1)
    branch1 = Conv2D(filters=16, kernel_size=(5, 5), strides=(1, 1), padding='same')(branch1)
    branch1 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(branch1)
    
    branch2 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same')(input_layer)
    branch2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same')(branch2)
    branch2 = Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same')(branch2)
    branch2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(branch2)
    
    branch3 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same')(input_layer)
    branch3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same')(branch3)
    branch3 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same')(branch3)
    branch3 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(branch3)
    
    # Concatenate branches
    x = Concatenate()([branch1, branch2, branch3])
    
    # Second block
    x = Conv2D(filters=128, kernel_size=(1, 1), strides=(1, 1), padding='same')(x)
    x = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
    x = Conv2D(filters=128, kernel_size=(5, 5), strides=(1, 1), padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(x)
    
    x = Conv2D(filters=256, kernel_size=(1, 1), strides=(1, 1), padding='same')(x)
    x = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
    x = Conv2D(filters=256, kernel_size=(5, 5), strides=(1, 1), padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(x)
    
    x = Conv2D(filters=512, kernel_size=(1, 1), strides=(1, 1), padding='same')(x)
    x = Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
    x = Conv2D(filters=512, kernel_size=(5, 5), strides=(1, 1), padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(x)
    
    # Flatten and dense layers
    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    x = Dense(10, activation='softmax')(x)
    
    # Create model
    model = keras.Model(inputs=input_layer, outputs=x)
    
    return model