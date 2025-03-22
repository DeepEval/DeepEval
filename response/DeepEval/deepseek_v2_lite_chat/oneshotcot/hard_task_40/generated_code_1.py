import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Concatenate, BatchNormalization, Reshape

def dl_model():
    input_layer = Input(shape=(28, 28, 1))
    
    # First block: process through three average pooling layers
    def block1(input_tensor):
        pool1 = MaxPooling2D(pool_size=(1, 1), strides=1, padding='valid')(input_tensor)
        pool2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(pool1)
        pool3 = MaxPooling2D(pool_size=(4, 4), strides=4, padding='valid')(pool2)
        return Flatten()(Concatenate()([pool3]))
    
    # Second block: four parallel paths with various convolutions and pooling
    def block2(input_tensor):
        path1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(path1)
        path3 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(path2)
        path4 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(path3)
        path4 = MaxPooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(path4)
        path4 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(path4)
        path4 = Flatten()(path4)
        path4 = Dense(units=128, activation='relu')(path4)
        path4 = Dense(units=64, activation='relu')(path4)
        return Flatten()(Concatenate()([path1, path2, path3, path4]))
    
    # Connect the blocks
    model = keras.Model(inputs=input_layer, outputs=block2(block1(input_tensor=input_layer)))
    
    return model