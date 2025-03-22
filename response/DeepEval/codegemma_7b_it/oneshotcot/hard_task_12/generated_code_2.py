import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, add

def dl_model():
    input_layer = Input(shape=(32, 32, 64))
    
    # Main Path
    main_path = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='valid', activation='sigmoid')(input_layer)
    main_path = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(main_path)
    main_path = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(main_path)
    
    # Branch Path
    branch_path = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='sigmoid')(input_layer)
    branch_path = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(branch_path)
    
    # Combine Paths
    combined_path = add([main_path, branch_path])
    
    # Output Layer
    flatten_layer = Flatten()(combined_path)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model