import keras
from keras.layers import Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Reshape, Dense, Multiply, Add
from keras.layers import Concatenate, BatchNormalization, Flatten

def dl_model():     

    input_layer = Input(shape=(32, 32, 3))
    
    # Main path
    main_path = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    main_path = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(main_path)
    main_path = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(main_path)
    main_path = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(main_path)
    
    # Branch path
    branch_path = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    branch_path = GlobalAveragePooling2D()(branch_path)
    branch_path = Dense(units=32, activation='relu')(branch_path)
    
    # Channel weights
    channel_weights = Dense(units=3 * 32, activation='softmax')(branch_path)
    channel_weights = Reshape((3, 32))(channel_weights)
    
    # Multiply channel weights with main path
    multiplied_path = Multiply()([main_path, channel_weights])
    
    # Add outputs from main and branch paths
    added_path = Add()([main_path, multiplied_path])
    
    # Classification
    bath_norm = BatchNormalization()(added_path)
    flatten_layer = Flatten()(bath_norm)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model