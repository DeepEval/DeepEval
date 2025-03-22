import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, GlobalAveragePooling2D, Flatten, Dense
from keras.models import Model

def dl_model():  

    input_layer = Input(shape=(32, 32, 3))

    # First Block
    conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv3 = Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(input_layer)
    maxpool = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(input_layer)
    
    concatenated = Concatenate()([conv1, conv2, conv3, maxpool])

    # Second Block
    pool = GlobalAveragePooling2D()(concatenated)
    dense1 = Dense(units=concatenated.shape[1], activation='relu')(pool) 
    dense2 = Dense(units=concatenated.shape[1], activation='relu')(dense1)
    
    # Reshape and multiply weights
    weights = Reshape((32, 32, concatenated.shape[1]))(dense2)
    elementwise_product = Multiply()([concatenated, weights]) 

    output_layer = Dense(units=10, activation='softmax')(elementwise_product)

    model = Model(inputs=input_layer, outputs=output_layer) 
    
    return model