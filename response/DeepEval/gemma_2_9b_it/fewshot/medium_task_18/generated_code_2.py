import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, Flatten, Dense

def dl_model():     

    input_layer = Input(shape=(32, 32, 3))
    
    # 1x1 Convolutions
    conv1_1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), activation='relu')(input_layer)
    
    # 3x3 Convolutions
    conv1_2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), activation='relu')(input_layer)
    
    # 5x5 Convolutions
    conv1_3 = Conv2D(filters=128, kernel_size=(5, 5), strides=(1, 1), activation='relu')(input_layer)
    
    # Max Pooling
    pool1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(input_layer)

    # Combine Features
    concat_layer = Concatenate()([conv1_1, conv1_2, conv1_3, pool1])
    
    # Flatten and Fully Connected Layers
    flatten_layer = Flatten()(concat_layer)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model