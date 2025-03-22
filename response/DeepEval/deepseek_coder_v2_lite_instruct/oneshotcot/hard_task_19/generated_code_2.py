import keras
from keras.layers import Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, Add, Multiply, BatchNormalization, Flatten
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # Main path
    main_path = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    main_path = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(main_path)
    main_path = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(main_path)
    main_path = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(main_path)
    
    # Branch path
    branch_path = GlobalAveragePooling2D()(main_path)
    branch_path = Dense(units=64, activation='relu')(branch_path)
    branch_path = Dense(units=32, activation='relu')(branch_path)
    branch_weights = Dense(units=32, activation='softmax')(branch_path)
    branch_weights = keras.backend.reshape(branch_weights, (-1, 32, 1, 1))
    
    # Apply weights to the input
    weighted_input = Multiply()([main_path, branch_weights])
    
    # Add the outputs from both paths
    added_output = Add()([main_path, weighted_input])
    
    # Flatten the result
    flattened_output = Flatten()(added_output)
    
    # Additional fully connected layers
    dense1 = Dense(units=128, activation='relu')(flattened_output)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model