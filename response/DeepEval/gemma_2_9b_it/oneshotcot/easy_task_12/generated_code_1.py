import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():  
    
    input_layer = Input(shape=(28, 28, 1))
    
    # Main Path
    x = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    x = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(x) 

    x_branch = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)  
    x_branch = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x_branch)
    x_branch = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(x_branch)
   
    x = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(x)

    x = Concatenate()([x, x_branch])

    x = Flatten()(x)
    output_layer = Dense(units=10, activation='softmax')(x)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer) 
    return model