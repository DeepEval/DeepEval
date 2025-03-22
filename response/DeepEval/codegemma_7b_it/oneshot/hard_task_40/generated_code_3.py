import keras
from keras.layers import Input, AveragePooling2D, Concatenate, Flatten, Dense, Dropout, Reshape
from keras.regularizers import l2

def dl_model():

    input_layer = Input(shape=(28, 28, 1))
    
    # First block
    pool1 = AveragePooling2D(pool_size=(1, 1), strides=1, padding='valid')(input_layer)
    pool2 = AveragePooling2D(pool_size=(2, 2), strides=2, padding='valid')(input_layer)
    pool3 = AveragePooling2D(pool_size=(4, 4), strides=4, padding='valid')(input_layer)
    flatten_pool = Concatenate()([Flatten()(pool1), Flatten()(pool2), Flatten()(pool3)])
    dense_pool = Dense(units=128, activation='relu')(flatten_pool)
    reshape_pool = Reshape((1, 1, 128))(dense_pool)
    
    # Second block
    path1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(reshape_pool)
    path2 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(reshape_pool)
    path2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(path2)
    path3 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(reshape_pool)
    path3 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(path3)
    path4 = AveragePooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(reshape_pool)
    path4 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(path4)
    path1 = Dropout(rate=0.1)(path1)
    path2 = Dropout(rate=0.1)(path2)
    path3 = Dropout(rate=0.1)(path3)
    path4 = Dropout(rate=0.1)(path4)
    
    # Concatenate paths
    concat = Concatenate()([path1, path2, path3, path4])
    
    # Output layers
    dense1 = Dense(units=128, activation='relu', kernel_regularizer=l2(0.01))(concat)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model