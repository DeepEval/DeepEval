import keras
from keras.layers import Input, Conv2D, Add, Concatenate, BatchNormalization, Flatten, Dense
from keras import regularizers
from keras import backend as K

def dl_model():     

    input_layer = Input(shape=(32, 32, 3))
    
    # Main pathway
    conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Parallel branch
    conv1_1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv1_2 = Conv2D(filters=32, kernel_size=(1, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv1_3 = Conv2D(filters=32, kernel_size=(3, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    concat = Concatenate()([conv1_1, conv1_2, conv1_3])
    conv1_main = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(concat)
    
    # Additive connection
    add = Add()([conv1, conv1_main])
    
    # Batch normalization
    bath_norm = BatchNormalization()(add)
    
    # Convolution for feature extraction
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu')(bath_norm)
    
    # Convolution for feature extraction
    conv3 = Conv2D(filters=128, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu')(conv2)
    
    # Flatten
    flatten_layer = Flatten()(conv3)
    
    # First fully connected layer
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    
    # Second fully connected layer
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model