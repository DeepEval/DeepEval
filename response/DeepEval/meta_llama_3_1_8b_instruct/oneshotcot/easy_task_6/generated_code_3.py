import keras
from keras.layers import Input, Conv2D, Add, BatchNormalization, Flatten, Dense

def dl_model():     

    input_layer = Input(shape=(28, 28, 1))
    # Main path: Increase feature map width and then restore channels
    conv1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(2, 2), padding='valid', activation='relu')(input_layer)
    conv2 = Conv2D(filters=1, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
    
    # Branch path: Directly connect to the input
    branch = Conv2D(filters=1, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Combine two paths through addition
    combined = Add()([conv2, branch])
    
    # Batch normalization and flatten
    bath_norm = BatchNormalization()(combined)
    flatten_layer = Flatten()(bath_norm)
    
    # Dense layer
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model