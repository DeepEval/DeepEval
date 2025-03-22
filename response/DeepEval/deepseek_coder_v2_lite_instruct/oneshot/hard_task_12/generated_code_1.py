import keras
from keras.layers import Input, Conv2D, Add, BatchNormalization, Flatten, Dense
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 64))
    
    # Main path
    main_path = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    path1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(main_path)
    path2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(main_path)
    concatenated = keras.layers.concatenate([path1, path2])
    
    # Branch path
    branch_path = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Addition of main and branch paths
    added = Add()([concatenated, branch_path])
    
    # Batch normalization and flattening
    batch_norm = BatchNormalization()(added)
    flatten_layer = Flatten()(batch_norm)
    
    # Fully connected layers
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model