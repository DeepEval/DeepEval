import keras
from keras.layers import Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, Multiply, Add, BatchNormalization, Flatten
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # Main path
    main_path = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    main_path = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(main_path)
    main_path = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(main_path)
    main_path = MaxPooling2D(pool_size=(2, 2), strides=2)(main_path)
    
    # Branch path
    branch_path = GlobalAveragePooling2D()(main_path)
    branch_path = Dense(128, activation='relu')(branch_path)
    branch_path = Dense(64, activation='relu')(branch_path)
    branch_weights = Dense(32, activation='softmax')(branch_path)
    branch_weights = keras.layers.Reshape((1, 1, 32))(branch_weights)
    
    # Multiply branch weights with the main path
    weighted_main_path = Multiply()([main_path, branch_weights])
    
    # Add the main path and the weighted branch path
    added_output = Add()([main_path, weighted_main_path])
    
    # Flatten and add more fully connected layers
    flatten_layer = Flatten()(added_output)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model