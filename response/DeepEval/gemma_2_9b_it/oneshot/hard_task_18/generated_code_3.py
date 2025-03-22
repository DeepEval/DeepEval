import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Add, GlobalAveragePooling2D, Flatten, Dense

def dl_model():     

    input_layer = Input(shape=(32, 32, 3))
    
    # First Block
    conv1 = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv2 = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
    pool = AveragePooling2D(pool_size=(2, 2), strides=2, padding='same')(conv2)

    # Second Block
    main_path = GlobalAveragePooling2D()(conv2)
    main_path = Dense(units=16, activation='relu')(main_path)
    main_path = Dense(units=16, activation='relu')(main_path)
    
    # reshape for element-wise multiplication
    main_path = keras.layers.Reshape((16, 1))(main_path) 
    output = input_layer * main_path 
    
    # Flatten and Classification
    flatten_layer = Flatten()(output)
    dense3 = Dense(units=10, activation='softmax')(flatten_layer)
    
    model = keras.Model(inputs=input_layer, outputs=dense3)

    return model